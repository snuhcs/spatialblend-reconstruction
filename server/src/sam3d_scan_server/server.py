# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""
Distributed scan server: SAM3 (Segment) and SAM3D-Object (Reconstruct) as independent worker pools.

Originally designed and implemented by Dongho Han (Seoul National University) as
server_distributed.py. This module keeps its worker and servicer structure. In addition: the heavy imports
(torch, pytorch3d, transformers, sam3d_objects) happen when a worker loads its
model, not at import time, so everything else can run and be tested on a CPU-only machine; paths, GPUs and the
CUDA arch come from ServerConfig; readiness is reported through the standard gRPC health service; a worker that
cannot load fails its requests instead of hanging them; and shutdown is clean.
"""
from __future__ import annotations

import io
import json
import logging
import queue
import sys
import tempfile
import threading
import time
import traceback
from concurrent import futures
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import grpc
import numpy as np
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from PIL import Image, ImageDraw

from . import protocol
from .config import SAM3_REPO_ID, ServerConfig
from .prompt import Prompt, parse_prompt, sam3_prompt_kwargs, single_mask

pb, pb_grpc = protocol.load()
log = logging.getLogger("sam3d_scan_server")

SERVICE_NAME = "spatialblend.scan.v1.ReconstructionService"

# The two worker pools load their models on their own threads. Their first imports (torch, torchvision via transformers
# and via sam3d_objects) must not run at the same time: two threads importing torchvision at once can see it
# partially initialised ("cannot import name 'InterpolationMode' ... circular import"), which can fail the SAM3 pool's
# load. Model loads take this lock, so imports and weight loading happen one pool at a time.
_MODEL_LOAD_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def draw_points_on_image(image: Image.Image, points: List[List[int]], labels: List[int],
                         box: Optional[List[float]] = None) -> Image.Image:
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    if box is not None:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="yellow", width=3)
    radius = 10
    for (x, y), label in zip(points, labels):
        color = "green" if label == 1 else "red"
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                     fill=color, outline="white", width=2)
    return draw_img


def overlay_mask_on_image(image: Image.Image, mask_np: np.ndarray,
                          points: List, labels: List, box: Optional[List[float]] = None) -> Image.Image:
    try:
        image = image.convert("RGBA")
        h, w = mask_np.shape
        colored_mask = np.zeros((h, w, 4), dtype=np.uint8)
        colored_mask[mask_np == 1] = [30, 144, 255, 128]
        overlay_layer = Image.fromarray(colored_mask, mode="RGBA")
        if overlay_layer.size != image.size:
            overlay_layer = overlay_layer.resize(image.size, Image.NEAREST)
        combined = Image.alpha_composite(image, overlay_layer).convert("RGB")
        return draw_points_on_image(combined, points, labels, box)
    except Exception as e:
        log.warning("Overlay error: %s", e)
        return image


def valid_points_in_bounds(points, width: int, height: int):
    """The click points inside the image, as ([x, y] list, label list). Every click is a positive label.
    Kept for callers that only send clicks; the Segment path uses parse_prompt (which reads SegPoint.z)."""
    valid_points, valid_labels = [], []
    for p in points:
        if 0 <= p.x < width and 0 <= p.y < height:
            valid_points.append([p.x, p.y])
            valid_labels.append(1)
    return valid_points, valid_labels


def build_inference(inference_cls, config_path: str, device):
    """Constructs sam3d_objects' Inference. Newer checkouts take device=, the pinned one does not and uses the
    default CUDA device, which is cuda:0 of CUDA_VISIBLE_DEVICES here."""
    import inspect
    params = inspect.signature(inference_cls.__init__).parameters
    if "device" in params:
        return inference_cls(config_path, device=device, compile=False)
    return inference_cls(config_path, compile=False)


def format_transform_to_json(rotation, scale, translation, filename="reconstruction"):
    """The objinfo schema the clients read: obj_2_pt3d_transform.{scale, rotationQuaternion, translation, matrix}."""
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    def rotmat_to_4x4(rotmat):
        m = torch.eye(4, dtype=torch.float32)
        m[:3, :3] = torch.from_numpy(rotmat)
        return m

    rot_list = rotation.squeeze().tolist()
    scale_list = scale.squeeze().tolist()
    trans_list = translation.squeeze().tolist()
    matrix_list = []

    try:
        rot_zup_2_yup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        rot_yup_2_zup = rot_zup_2_yup.T
        yup_2_zup_transform = rotmat_to_4x4(rot_yup_2_zup)
        R_obj_2_pt3d = quaternion_to_matrix(rotation)

        obj_2_pt3d_zup = compose_transform(
            scale=scale,
            rotation=R_obj_2_pt3d,
            translation=translation,
        )
        obj_2_pt3d_transform = yup_2_zup_transform @ obj_2_pt3d_zup.get_matrix()[0]
        matrix_tensor = obj_2_pt3d_transform.T
        matrix_list = matrix_tensor.flatten().tolist()
    except Exception as e:
        log.warning("Matrix calculation failed: %s", e)
        traceback.print_exc()

    return {
        "filename": filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "obj_2_pt3d_transform": {
            "scale": scale_list,
            "rotationQuaternion": rot_list,
            "translation": trans_list,
            "matrix": matrix_list,
        },
    }


# ---------------------------------------------------------------------------
# Sam3Worker: SAM3 segmentation only
# ---------------------------------------------------------------------------

class Sam3Worker(threading.Thread):
    def __init__(self, gpu_id, request_queue, response_queues, cfg: ServerConfig):
        super().__init__(name=f"sam3-gpu{gpu_id}", daemon=True)
        self.gpu_id = gpu_id
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.cfg = cfg
        self.device = None
        self.sam3_model = None
        self.sam3_processor = None
        self.model_ready = True      # False after a failed load
        self.ready = threading.Event()  # set once the model is loaded

    def _load_models(self):
        with _MODEL_LOAD_LOCK:
            self._load_models_locked()

    def _load_models_locked(self):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self.device = torch.device(f"cuda:{self.gpu_id}")
        log.info("[Sam3Worker-GPU%s] Loading SAM3 model...", self.gpu_id)
        try:
            self.sam3_model = Sam3TrackerModel.from_pretrained(SAM3_REPO_ID).to(self.device)
            self.sam3_processor = Sam3TrackerProcessor.from_pretrained(SAM3_REPO_ID)
            log.info("[Sam3Worker-GPU%s] SAM3 model ready.", self.gpu_id)
            self.ready.set()
        except Exception as e:
            log.error("[Sam3Worker-GPU%s] Failed to load SAM3 model: %s", self.gpu_id, e)
            traceback.print_exc()
            self.model_ready = False

    def _run_sam3_inference(self, image: Image.Image, prompt: Prompt) -> np.ndarray:
        import torch

        inputs = self.sam3_processor(
            images=image,
            return_tensors="pt",
            **sam3_prompt_kwargs(prompt),
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam3_model(**inputs, multimask_output=not single_mask(prompt))

        masks_result = self.sam3_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )
        best_mask_idx = torch.argmax(outputs.iou_scores[0, 0]).item()
        best_mask = masks_result[0][0][best_mask_idx]
        # Return H*W uint8 array, values 0 or 1
        return best_mask.numpy().astype(np.uint8)

    def run(self):
        self._load_models()
        log.info("[Sam3Worker-GPU%s] Standing by.", self.gpu_id)

        while self.model_ready:
            req_id, client_id, image_data, points = self.request_queue.get()

            try:
                log.info("[Sam3Worker-GPU%s] Processing Segment from '%s'", self.gpu_id, client_id)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                width, height = image.size
                log.info("[Sam3Worker-GPU%s]   Image: %sx%s", self.gpu_id, width, height)

                prompt = parse_prompt(points, width, height)
                log.info("[Sam3Worker-GPU%s]   Prompt: %s (of %s sent)", self.gpu_id, prompt.summary(), len(points))

                t0 = time.time()
                mask = self._run_sam3_inference(image, prompt)
                elapsed = time.time() - t0
                log.info("[Sam3Worker-GPU%s]   SAM3 done (%.3fs)", self.gpu_id, elapsed)

                if self.cfg.debug_dir is not None:
                    self.cfg.debug_dir.mkdir(parents=True, exist_ok=True)
                    overlay = overlay_mask_on_image(image, mask, prompt.points, prompt.labels, prompt.box)
                    debug_file = self.cfg.debug_dir / f"seg_debug_{client_id}_{time.time():.0f}.png"
                    overlay.save(debug_file)
                    log.info("[Sam3Worker-GPU%s]   Saved debug overlay: %s", self.gpu_id, debug_file)

                h, w = mask.shape
                result = {
                    "success": True,
                    "message": (
                        f"Segment OK | GPU:{self.gpu_id} | "
                        f"{prompt.summary()} | time:{elapsed:.3f}s"
                    ),
                    "mask_data": mask.tobytes(),
                    "mask_width": w,
                    "mask_height": h,
                }

            except Exception as e:
                log.error("[Sam3Worker-GPU%s] Error: %s", self.gpu_id, e)
                traceback.print_exc()
                result = {
                    "success": False,
                    "message": str(e),
                    "mask_data": b"",
                    "mask_width": 0,
                    "mask_height": 0,
                }

            if req_id in self.response_queues:
                self.response_queues[req_id].put(result)
            self.request_queue.task_done()


# ---------------------------------------------------------------------------
# Sam3dObjectWorker: SAM3D-Object reconstruction only
# ---------------------------------------------------------------------------

class Sam3dObjectWorker(threading.Thread):
    def __init__(self, gpu_id, request_queue, response_queues, cfg: ServerConfig):
        super().__init__(name=f"sam3d-gpu{gpu_id}", daemon=True)
        self.gpu_id = gpu_id
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.cfg = cfg
        self.device = None
        self.sam3d_objects_model = None
        self.model_ready = True
        self.ready = threading.Event()

    def _load_models(self):
        with _MODEL_LOAD_LOCK:
            self._load_models_locked()

    def _load_models_locked(self):
        import torch

        self.device = torch.device(f"cuda:{self.gpu_id}")
        log.info("[Sam3dObjWorker-GPU%s] Loading SAM3D-Object model...", self.gpu_id)
        try:
            config_path = self.cfg.sam3d_pipeline_yaml
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            notebook = str(Path(self.cfg.sam3d_root) / "notebook")
            if notebook not in sys.path:
                sys.path.append(notebook)
            from inference import Inference  # sam-3d-objects/notebook/inference.py
            self.sam3d_objects_model = build_inference(Inference, str(config_path), self.device)
            log.info("[Sam3dObjWorker-GPU%s] SAM3D-Object model ready.", self.gpu_id)
            self.ready.set()
        except Exception as e:
            log.error("[Sam3dObjWorker-GPU%s] Failed to load model: %s", self.gpu_id, e)
            traceback.print_exc()
            self.model_ready = False

    def run(self):
        self._load_models()
        log.info("[Sam3dObjWorker-GPU%s] Standing by.", self.gpu_id)

        while self.model_ready:
            req_id, client_id, image_data, mask_data, mask_width, mask_height = self.request_queue.get()

            try:
                log.info("[Sam3dObjWorker-GPU%s] Processing Reconstruct from '%s'", self.gpu_id, client_id)

                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image_np = np.array(image, dtype=np.uint8)

                # Decode raw mask bytes
                mask_np = np.frombuffer(mask_data, dtype=np.uint8).reshape(mask_height, mask_width)
                log.info("[Sam3dObjWorker-GPU%s]   Mask: %sx%s, foreground pixels: %s",
                         self.gpu_id, mask_width, mask_height, mask_np.sum())

                t0 = time.time()
                result = self.sam3d_objects_model(image_np, mask_np, seed=42)
                elapsed = time.time() - t0
                log.info("[Sam3dObjWorker-GPU%s]   Reconstruction done (%.3fs)", self.gpu_id, elapsed)

                if "glb" not in result or result["glb"] is None:
                    raise RuntimeError("3D Reconstruction failed to produce GLB output.")

                with tempfile.NamedTemporaryFile(suffix=".glb", delete=True) as tmp:
                    result["glb"].export(tmp.name)
                    tmp.seek(0)
                    glb_bytes = tmp.read()

                rot = result["rotation"].cpu()
                scale = result["scale"].cpu()
                translation = result["translation"].cpu()
                obj_info = format_transform_to_json(rot, scale, translation, "reconstruction")
                json_bytes = json.dumps(obj_info).encode("utf-8")

                response = {
                    "success": True,
                    "message": (
                        f"Reconstruct OK | GPU:{self.gpu_id} | "
                        f"glb:{len(glb_bytes)}B | json:{len(json_bytes)}B | "
                        f"time:{elapsed:.3f}s"
                    ),
                    "glb_data": glb_bytes,
                    "json_data": json_bytes,
                }

            except Exception as e:
                log.error("[Sam3dObjWorker-GPU%s] Error: %s", self.gpu_id, e)
                traceback.print_exc()
                response = {
                    "success": False,
                    "message": str(e),
                    "glb_data": b"",
                    "json_data": b"",
                }

            if req_id in self.response_queues:
                self.response_queues[req_id].put(response)
            self.request_queue.task_done()


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------

WorkerFactory = Callable[[int, "queue.Queue", dict, ServerConfig], threading.Thread]


class ReconstructionServicer(pb_grpc.ReconstructionServiceServicer):
    def __init__(self, cfg: ServerConfig, sam3_gpu_ids: Optional[List[int]] = None,
                 sam3d_object_gpu_ids: Optional[List[int]] = None,
                 sam3_factory: WorkerFactory = Sam3Worker,
                 sam3d_object_factory: WorkerFactory = Sam3dObjectWorker):
        self.cfg = cfg
        self.sam3_queue = queue.Queue()
        self.sam3d_object_queue = queue.Queue()
        self.response_queues = {}
        self.lock = threading.Lock()
        # Workers address the devices left visible by CUDA_VISIBLE_DEVICES: 0..len(gpus)-1.
        local_ids = list(range(len(cfg.gpus))) or [0]
        sam3_gpu_ids = local_ids if sam3_gpu_ids is None else sam3_gpu_ids
        sam3d_object_gpu_ids = local_ids if sam3d_object_gpu_ids is None else sam3d_object_gpu_ids

        self.sam3_workers = []
        for gid in sam3_gpu_ids:
            w = sam3_factory(gid, self.sam3_queue, self.response_queues, cfg)
            w.start()
            self.sam3_workers.append(w)

        self.sam3d_object_workers = []
        for gid in sam3d_object_gpu_ids:
            w = sam3d_object_factory(gid, self.sam3d_object_queue, self.response_queues, cfg)
            w.start()
            self.sam3d_object_workers.append(w)

        log.info("SAM3 workers on devices: %s", sam3_gpu_ids)
        log.info("SAM3D-Object workers on devices: %s", sam3d_object_gpu_ids)

    # readiness ---------------------------------------------------------------

    @staticmethod
    def _pool_state(workers):
        """(loaded, failed, total)."""
        loaded = sum(1 for w in workers if w.ready.is_set())
        failed = sum(1 for w in workers if not w.model_ready)
        return loaded, failed, len(workers)

    def readiness(self) -> dict:
        s_loaded, s_failed, s_total = self._pool_state(self.sam3_workers)
        o_loaded, o_failed, o_total = self._pool_state(self.sam3d_object_workers)
        return {
            "segment": {"loaded": s_loaded, "failed": s_failed, "total": s_total},
            "reconstruct": {"loaded": o_loaded, "failed": o_failed, "total": o_total},
            "serving": s_loaded > 0 and o_loaded > 0,
        }

    # RPCs --------------------------------------------------------------------

    def _make_req_id(self, client_id: str) -> str:
        return f"{client_id}_{time.time()}"

    def _dispatch(self, work_queue, workers, item_tail, req_id: str, what: str):
        """Queue one job and wait for its result. A pool with no loaded worker fails the request at once."""
        if not any(w.model_ready for w in workers):
            return {"success": False, "message": f"{what}: no worker has a loaded model on this server"}
        resp_q = queue.Queue()
        with self.lock:
            self.response_queues[req_id] = resp_q
        try:
            work_queue.put((req_id,) + item_tail)
            try:
                return resp_q.get(timeout=self.cfg.request_timeout_s)
            except queue.Empty:
                return {"success": False,
                        "message": f"{what}: no result within {self.cfg.request_timeout_s:.0f} s"}
        finally:
            with self.lock:
                self.response_queues.pop(req_id, None)

    def Segment(self, request, context):
        req_id = self._make_req_id(request.client_id)
        log.info("Segment request from '%s'", request.client_id)
        result = self._dispatch(self.sam3_queue, self.sam3_workers,
                                (request.client_id, request.image_data, request.points), req_id, "Segment")
        return pb.SegmentResponse(
            success=result["success"],
            message=result["message"],
            mask_data=result.get("mask_data", b""),
            mask_width=result.get("mask_width", 0),
            mask_height=result.get("mask_height", 0),
        )

    def Reconstruct(self, request, context):
        req_id = self._make_req_id(request.client_id)
        log.info("Reconstruct request from '%s'", request.client_id)
        result = self._dispatch(
            self.sam3d_object_queue, self.sam3d_object_workers,
            (request.client_id, request.image_data, request.mask_data, request.mask_width, request.mask_height),
            req_id, "Reconstruct")
        return pb.ReconstructResponse(
            success=result["success"],
            message=result["message"],
            glb_file=result.get("glb_data", b""),
            json_file=result.get("json_data", b""),
        )


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

class ScanServer:
    """A running gRPC server plus its health service. stop() is idempotent."""

    def __init__(self, cfg: ServerConfig, servicer: pb_grpc.ReconstructionServiceServicer):
        self.cfg = cfg
        self.servicer = servicer
        self._stop = threading.Event()
        opts = [("grpc.max_receive_message_length", cfg.max_message_mb * 1024 * 1024),
                ("grpc.max_send_message_length", cfg.max_message_mb * 1024 * 1024)]
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg.max_workers), options=opts)
        pb_grpc.add_ReconstructionServiceServicer_to_server(servicer, self.server)
        self.health = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(self.health, self.server)
        self._set_health(health_pb2.HealthCheckResponse.NOT_SERVING)
        self.port = self.server.add_insecure_port(f"{cfg.host}:{cfg.port}")
        if self.port == 0:
            raise OSError(f"cannot bind {cfg.host}:{cfg.port}")
        self._watcher = threading.Thread(target=self._watch_readiness, name="health-watch", daemon=True)

    def _set_health(self, status):
        for name in ("", SERVICE_NAME):
            self.health.set(name, status)

    def _serving(self) -> bool:
        readiness = getattr(self.servicer, "readiness", None)
        return True if readiness is None else bool(readiness()["serving"])

    def _watch_readiness(self):
        last = None
        while not self._stop.is_set():
            status = (health_pb2.HealthCheckResponse.SERVING if self._serving()
                      else health_pb2.HealthCheckResponse.NOT_SERVING)
            if status != last:
                self._set_health(status)
                log.info("health: %s", health_pb2.HealthCheckResponse.ServingStatus.Name(status))
                last = status
            self._stop.wait(1.0)

    def start(self):
        self.server.start()
        self._watcher.start()
        log.info("--- Distributed server started on port %s ---", self.port)

    def wait(self):
        self.server.wait_for_termination()

    def stop(self, grace: float = 5.0):
        if self._stop.is_set():
            return
        self._stop.set()
        self._set_health(health_pb2.HealthCheckResponse.NOT_SERVING)
        self.server.stop(grace).wait(grace + 5.0)


def build_server(cfg: ServerConfig) -> ScanServer:
    """Real workers, or the deterministic stub with cfg.stub."""
    if cfg.stub:
        from .stub import Servicer as StubServicer
        return ScanServer(cfg, StubServicer())
    return ScanServer(cfg, ReconstructionServicer(cfg))
