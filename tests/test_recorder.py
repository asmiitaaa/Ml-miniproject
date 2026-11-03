"""Regression checks for asynchronous screenshot files observed on Windows."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("record_demo", ROOT / "scripts/record_demo.py")
RECORDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECORDER)
RUNNING_PROCESS = SimpleNamespace(poll=lambda: None)


def test_incomplete_png_is_not_accepted_as_a_recording_frame(tmp_path):
    frame = tmp_path / "in_progress.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(RuntimeError, match="did not save"):
        RECORDER.wait_for_file(frame, RUNNING_PROCESS, timeout=.02)


def test_complete_png_is_verified_and_hashes_actual_bytes(tmp_path):
    frame = tmp_path / "complete.png"
    Image.new("RGB", (8, 8), "#123456").save(frame)
    digest = RECORDER.wait_for_file(frame, RUNNING_PROCESS, timeout=1)
    assert digest == hashlib.sha256(frame.read_bytes()).hexdigest()
    assert digest != hashlib.sha256(b"").hexdigest()


def test_misaligned_controller_frames_are_not_composed(tmp_path):
    args = SimpleNamespace(width=160, height=120, seed=11, scenario="test", no_video=True)
    records = [{"controller": "fixed", "frames": [{"simulation_clock_s": 5}]},
               {"controller": "actuated", "frames": []}]
    with pytest.raises(ValueError, match="different or empty"):
        RECORDER.compose(tmp_path, records, args, "TEST")
