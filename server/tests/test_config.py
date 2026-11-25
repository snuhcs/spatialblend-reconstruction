# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Configuration: defaults are machine-neutral, the environment sets every field, and a flag wins over the environment."""
from pathlib import Path

import pytest

from sam3d_scan_server import cli
from sam3d_scan_server.config import ServerConfig, env_defaults, parse_gpus


def args(*argv):
    p = cli.argparse.ArgumentParser()
    cli._serve_parser(p)
    return p.parse_args(list(argv))


def test_defaults_bind_loopback_on_gpu_0():
    cfg = cli.build_config(args(), env={})
    assert (cfg.host, cfg.port, cfg.gpus, cfg.sam3d_root, cfg.stub) == ("127.0.0.1", 9800, [0], None, False)


def test_environment_sets_every_field():
    env = {"SAM3D_SCAN_HOST": "0.0.0.0", "SAM3D_SCAN_PORT": "9811", "SAM3D_SCAN_GPUS": "2,3",
           "SAM3D_OBJECTS_ROOT": "/opt/sam3d", "SAM3D_OBJECTS_CHECKPOINTS": "/models/sam3d",
           "SAM3D_SCAN_DEBUG_DIR": "/tmp/dbg", "SAM3D_SCAN_MAX_MESSAGE_MB": "32",
           "SAM3D_SCAN_REQUEST_TIMEOUT_S": "120"}
    cfg = cli.build_config(args(), env=env)
    assert cfg.host == "0.0.0.0" and cfg.port == 9811 and cfg.gpus == [2, 3]
    assert cfg.sam3d_root == Path("/opt/sam3d") and cfg.sam3d_pipeline_yaml == Path("/models/sam3d/pipeline.yaml")
    assert cfg.debug_dir == Path("/tmp/dbg") and cfg.max_message_mb == 32 and cfg.request_timeout_s == 120.0


def test_flags_win_over_the_environment():
    env = {"SAM3D_SCAN_PORT": "9811", "SAM3D_SCAN_GPUS": "2", "SAM3D_SCAN_HOST": "0.0.0.0"}
    cfg = cli.build_config(args("--port", "9900", "--gpu", "5", "6"), env=env)
    assert cfg.port == 9900 and cfg.gpus == [5, 6] and cfg.host == "0.0.0.0"


def test_checkpoints_default_inside_the_checkout():
    assert ServerConfig(sam3d_root=Path("/x")).sam3d_pipeline_yaml == Path("/x/checkpoints/hf/pipeline.yaml")


@pytest.mark.parametrize("text, gpus", [("0", [0]), ("0,1", [0, 1]), (" 2 3 ", [2, 3])])
def test_parse_gpus(text, gpus):
    assert parse_gpus(text) == gpus


@pytest.mark.parametrize("env", [{"SAM3D_SCAN_PORT": "x"}, {"SAM3D_SCAN_GPUS": "a,b"}])
def test_malformed_environment_is_an_error(env):
    with pytest.raises(ValueError):
        env_defaults(env)


def test_missing_checkout_is_a_usage_error(monkeypatch, capsys):
    for k in ("SAM3D_OBJECTS_ROOT", "SAM3D_OBJECTS_CHECKPOINTS"):
        monkeypatch.delenv(k, raising=False)
    assert cli.main([]) == cli.EXIT_USAGE
    assert "--sam3d-root" in capsys.readouterr().err


def test_malformed_environment_is_ignored_when_a_flag_gives_the_field():
    assert cli.build_config(args("--port", "9900"), env={"SAM3D_SCAN_PORT": "x"}).port == 9900


def test_missing_checkpoints_is_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SAM3D_OBJECTS_CHECKPOINTS", raising=False)
    assert cli.main(["--sam3d-root", str(tmp_path)]) == cli.EXIT_USAGE
    assert "pipeline.yaml not found" in capsys.readouterr().err


def test_models_check_reports_without_reading_the_token(tmp_path, monkeypatch, capsys):
    from sam3d_scan_server import models
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "pipeline.yaml").write_text("")
    assert models.main(["--check", "--checkpoints", str(tmp_path / "ck")]) == 0
    out = capsys.readouterr().out
    assert "login: MISSING" in out and "checkpoints: present" in out


def test_models_refuses_to_replace_an_unrelated_directory(tmp_path, monkeypatch, capsys):
    from sam3d_scan_server import models
    monkeypatch.setenv("HF_TOKEN", "x")   # presence only: nothing is downloaded, the refusal comes first
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "keep.txt").write_text("mine")
    assert models.main(["--only", "sam3d-objects", "--checkpoints", str(shared)]) == 2
    assert (shared / "keep.txt").is_file() and "refusing" in capsys.readouterr().err


def _fake_hub(monkeypatch, fail_move=False):
    import sys, types
    from sam3d_scan_server import models

    def snapshot_download(repo_id, local_dir=None, **_):
        d = Path(local_dir) / "checkpoints"
        d.mkdir(parents=True)
        (d / "pipeline.yaml").write_text("new")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setenv("HF_TOKEN", "x")
    if fail_move:
        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(models.shutil, "move", boom)
    return models


def test_models_replaces_a_previous_install_and_leaves_siblings_alone(tmp_path, monkeypatch):
    models = _fake_hub(monkeypatch)
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "pipeline.yaml").write_text("old")
    for sibling in ("ck-old", "ck-download"):
        (tmp_path / sibling).mkdir()
        (tmp_path / sibling / "user.txt").write_text("mine")
    assert models.main(["--only", "sam3d-objects", "--checkpoints", str(ck)]) == 0
    assert (ck / "pipeline.yaml").read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ck", "ck-download", "ck-old"]
    assert (tmp_path / "ck-old" / "user.txt").is_file() and (tmp_path / "ck-download" / "user.txt").is_file()


def test_models_restores_the_previous_install_when_the_move_fails(tmp_path, monkeypatch):
    models = _fake_hub(monkeypatch, fail_move=True)
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "pipeline.yaml").write_text("old")
    assert models.main(["--only", "sam3d-objects", "--checkpoints", str(ck)]) == 1
    assert (ck / "pipeline.yaml").read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["ck"]
