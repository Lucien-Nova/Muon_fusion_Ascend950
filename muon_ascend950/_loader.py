from __future__ import annotations

import os
from pathlib import Path


_LOADED = False


def _library_path() -> Path:
    configured = os.environ.get("MUON_ASCEND950_LIBRARY")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"MUON_ASCEND950_LIBRARY does not exist: {path}")
        return path

    package_dir = Path(__file__).resolve().parent
    candidates = sorted(package_dir.glob("_C*.so"))
    if not candidates:
        raise RuntimeError(
            "muon_ascend950 native library is missing. Reinstall the wheel built "
            "for the current torch/torch_npu environment."
        )
    return candidates[0]


def load_library() -> None:
    global _LOADED
    if _LOADED:
        return

    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("muon_ascend950 requires torch and torch_npu") from exc

    torch.ops.load_library(str(_library_path()))

    def fake_polar_step(x, a: float, b: float, c: float):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    try:
        torch.library.register_fake("muon_ascend950::polar_step")(fake_polar_step)
    except RuntimeError as exc:
        if "already has" not in str(exc):
            raise
    _LOADED = True

