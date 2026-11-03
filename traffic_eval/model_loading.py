"""Load original SB3 archives through seek-safe in-memory tensor streams.

The current Windows PyTorch reader fails on ZipExtFile streams while the
identical bytes load correctly from BytesIO. Buffer only tensor members;
keep SB3's validation and weights_only=True deserialization unchanged.
"""
from __future__ import annotations

import io
from pathlib import Path
import threading
import types
import zipfile

_LOAD_LOCK = threading.RLock()


class _BufferedTensorZip(zipfile.ZipFile):
    def open(self, name, mode="r", pwd=None, *, force_zip64=False):
        stream = super().open(name, mode, pwd, force_zip64=force_zip64)
        member = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        if mode == "r" and member.endswith(".pth"):
            with stream:
                return io.BytesIO(stream.read())
        return stream


def load_dqn(checkpoint: str | Path):
    from stable_baselines3 import DQN
    from stable_baselines3.common import save_util

    # Scope the compatibility shim to SB3's own ZIP module reference. The
    # standard-library zipfile module and original checkpoint are untouched.
    with _LOAD_LOCK:
        original = save_util.zipfile
        proxy = types.SimpleNamespace(**vars(original))
        proxy.ZipFile = _BufferedTensorZip
        save_util.zipfile = proxy
        try:
            return DQN.load(checkpoint, device="cpu")
        finally:
            save_util.zipfile = original
