# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""
probe_distributed.py

Distributed pipeline test client:
  1. Load image & point list (matching pair from images/ and points/ dirs)
  2. Segment phase: call Segment RPC with 1 point, then 2, ..., then all points.
     Save overlay image for each step.
  3. Reconstruct phase: use the final mask (all points used) to call Reconstruct.
     Save GLB and JSON outputs.

Usage:
    # The generated proto modules are built on first use by sam3d_scan_server.protocol.
    # Bring your own images and point files: none ship with this repository.
    python probe_distributed.py \
        --name User1 \
        --image_dir images \
        --points_dir points \
        --addr localhost:9800

    # Test with specific image/points pair:
    python probe_distributed.py --name User1 \
        --image images/chair.png \
        --points points/chair.txt \
        --addr localhost:9800
"""

import grpc
from sam3d_scan_server import protocol

distributed_pb2, distributed_pb2_grpc = protocol.load()
import time
import argparse
import os
import sys
import random
import io
import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_points_from_file(filepath):
    """Load (x, y) points from a text file. Returns list of (x, y) tuples."""
    points = []
    if not os.path.exists(filepath):
        print(f"Warning: Points file not found: {filepath}")
        return points
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.replace(",", " ").strip().split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append((x, y))
        print(f"   Loaded {len(points)} points from {os.path.basename(filepath)}")
    except Exception as e:
        print(f"Error parsing points file: {e}")
    return points


def pick_image_and_points(image_dir, points_dir):
    """Pick a random image from image_dir and its matching points file."""
    valid_extensions = (".png", ".jpg", ".jpeg")
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    if not images:
        print(f"Error: No images found in '{image_dir}'.")
        sys.exit(1)

    selected = random.choice(images)
    image_path = os.path.join(image_dir, selected)
    base_name = os.path.splitext(selected)[0]
    points_path = os.path.join(points_dir, base_name + ".txt")
    return image_path, points_path


def points_to_proto(xy_list):
    """Convert list of (x, y) tuples to proto Point list."""
    return [distributed_pb2.SegPoint(x=x, y=y, z=0.0) for x, y in xy_list]


# ---------------------------------------------------------------------------
# Overlay helper (client-side, mirrors server debug logic)
# ---------------------------------------------------------------------------

def overlay_mask_on_image(image: Image.Image, mask_np: np.ndarray,
                          points, labels) -> Image.Image:
    """Overlay binary mask and point markers onto image."""
    image = image.convert("RGBA")
    h, w = mask_np.shape
    colored_mask = np.zeros((h, w, 4), dtype=np.uint8)
    colored_mask[mask_np == 1] = [30, 144, 255, 128]
    overlay_layer = Image.fromarray(colored_mask, mode="RGBA")
    if overlay_layer.size != image.size:
        overlay_layer = overlay_layer.resize(image.size, Image.NEAREST)
    combined = Image.alpha_composite(image, overlay_layer).convert("RGB")

    draw = ImageDraw.Draw(combined)
    radius = 10
    for (x, y), label in zip(points, labels):
        color = "green" if label == 1 else "red"
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                     fill=color, outline="white", width=2)
    return combined


# ---------------------------------------------------------------------------
# Segment phase: call N times with 1..N points
# ---------------------------------------------------------------------------

def run_segment_phase(stub, client_name, image_bytes, all_points, output_dir):
    """
    Call Segment RPC incrementally:
      step 1 -> points[:1]
      step 2 -> points[:2]
      ...
      step N -> points[:N]  (all points)

    Returns the final mask_data, mask_width, mask_height from the last call,
    or None on failure.
    """
    print(f"\n[{client_name}] === Segment Phase: {len(all_points)} steps ===")

    final_mask_data = None
    final_mask_width = 0
    final_mask_height = 0

    options = [("grpc.max_receive_message_length", 100 * 1024 * 1024)]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    for num_pts in range(1, len(all_points) + 1):
        pts_subset = all_points[:num_pts]
        proto_points = points_to_proto(pts_subset)

        print(f"[{client_name}] Segment step {num_pts}/{len(all_points)}: "
              f"{num_pts} point(s)...")

        request = distributed_pb2.SegmentRequest(
            client_id=client_name,
            image_data=image_bytes,
            points=proto_points,
        )

        t0 = time.time()
        try:
            response = stub.Segment(request)
        except grpc.RpcError as e:
            print(f"[{client_name}]   Segment gRPC error (step {num_pts}): {e.details()}")
            return None, 0, 0

        elapsed = time.time() - t0

        if not response.success:
            print(f"[{client_name}]   Segment failed (step {num_pts}): {response.message}")
            return None, 0, 0

        print(f"[{client_name}]   Segment OK ({elapsed:.2f}s) | "
              f"mask {response.mask_width}x{response.mask_height} | "
              f"{response.message}")

        # Decode mask and save overlay
        mask_np = np.frombuffer(response.mask_data, dtype=np.uint8).reshape(
            response.mask_height, response.mask_width
        )
        labels = [1] * len(pts_subset)
        overlay = overlay_mask_on_image(image, mask_np, pts_subset, labels)
        overlay_path = os.path.join(
            output_dir, f"overlay_{client_name}_step{num_pts:02d}.png"
        )
        overlay.save(overlay_path)
        print(f"[{client_name}]   Saved overlay: {overlay_path}")

        # Keep the last response for Reconstruct
        final_mask_data = response.mask_data
        final_mask_width = response.mask_width
        final_mask_height = response.mask_height

    print(f"\n[{client_name}] Segment phase complete. "
          f"Final mask: {final_mask_width}x{final_mask_height}, "
          f"{len(all_points)} points used.")
    return final_mask_data, final_mask_width, final_mask_height


# ---------------------------------------------------------------------------
# Reconstruct phase
# ---------------------------------------------------------------------------

def run_reconstruct_phase(stub, client_name, image_bytes,
                          mask_data, mask_width, mask_height, output_dir):
    """Call Reconstruct RPC with the final mask from Segment phase."""
    print(f"\n[{client_name}] === Reconstruct Phase ===")

    request = distributed_pb2.ReconstructRequest(
        client_id=client_name,
        image_data=image_bytes,
        mask_data=mask_data,
        mask_width=mask_width,
        mask_height=mask_height,
    )

    t0 = time.time()
    try:
        response = stub.Reconstruct(request)
    except grpc.RpcError as e:
        print(f"[{client_name}] Reconstruct gRPC error: {e.details()}")
        return False

    elapsed = time.time() - t0

    if not response.success:
        print(f"[{client_name}] Reconstruct failed: {response.message}")
        return False

    print(f"[{client_name}] Reconstruct OK ({elapsed:.2f}s) | {response.message}")

    glb_path = os.path.join(output_dir, f"result_{client_name}_objmodel.glb")
    json_path = os.path.join(output_dir, f"result_{client_name}_objinfo.json")

    with open(glb_path, "wb") as f:
        f.write(response.glb_file)
    print(f"[{client_name}] Saved GLB:  {glb_path}")

    if response.json_file:
        with open(json_path, "wb") as f:
            f.write(response.json_file)
        print(f"[{client_name}] Saved JSON: {json_path}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_client(client_name, image_path, points_path, server_address, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{client_name}] Image:  {image_path}")
    print(f"[{client_name}] Points: {points_path}")
    print(f"[{client_name}] Server: {server_address}")
    print(f"[{client_name}] Output: {output_dir}")

    # Load image bytes
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"[{client_name}] Failed to read image: {e}")
        return False

    # Load all points
    all_points = load_points_from_file(points_path)
    if not all_points:
        print(f"[{client_name}] No points loaded.")
        return False

    options = [("grpc.max_receive_message_length", 100 * 1024 * 1024),
               ("grpc.max_send_message_length", 100 * 1024 * 1024)]

    with grpc.insecure_channel(server_address, options=options) as channel:
        stub = distributed_pb2_grpc.ReconstructionServiceStub(channel)

        # Phase 1: Segment (1 point, 2 points, ..., all points)
        mask_data, mask_width, mask_height = run_segment_phase(
            stub, client_name, image_bytes, all_points, output_dir
        )
        if mask_data is None:
            print(f"[{client_name}] Segment phase failed. Aborting.")
            return False

        # Phase 2: Reconstruct with final mask
        success = run_reconstruct_phase(
            stub, client_name, image_bytes,
            mask_data, mask_width, mask_height, output_dir
        )

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="User1")
    parser.add_argument("--addr", type=str, default="localhost:9800")
    parser.add_argument("--output_dir", type=str, default="distributed_output",
                        help="Directory to save overlay images, GLB, JSON")

    # Image/points: either specify directly or let the script pick randomly
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image_dir", type=str, default=None,
                       help="Directory to pick a random image from")
    group.add_argument("--image", type=str, default=None,
                       help="Specific image file path")

    parser.add_argument("--points_dir", type=str, default=None,
                        help="Directory with point txt files (used with --image_dir)")
    parser.add_argument("--points", type=str, default=None,
                        help="Specific points txt file (used with --image)")

    args = parser.parse_args()

    # Resolve image and points paths
    if args.image:
        image_path = args.image
        if args.points:
            points_path = args.points
        else:
            # Infer points file from image name
            base = os.path.splitext(os.path.basename(image_path))[0]
            points_dir = args.points_dir or "points"
            points_path = os.path.join(points_dir, base + ".txt")
    else:
        image_dir = args.image_dir or "images"
        points_dir = args.points_dir or "points"
        image_path, points_path = pick_image_and_points(image_dir, points_dir)

    success = run_client(args.name, image_path, points_path, args.addr, args.output_dir)
    sys.exit(0 if success else 1)
