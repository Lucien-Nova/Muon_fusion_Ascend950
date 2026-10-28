from __future__ import annotations

import torch


def polar_step(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """Run one Polar Express iteration and return a tensor with the same shape.

    ``x`` must be a BF16 NPU tensor with shape ``[M,N]`` or ``[B,M,N]``.
    The wrapper handles ``M > N`` by transposing around the native operator.
    Non-aligned dimensions are padded and cropped inside the C++ binding.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type != "npu":
        raise ValueError(f"x must be on an NPU, got {x.device}")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must use torch.bfloat16, got {x.dtype}")
    if x.ndim not in (2, 3):
        raise ValueError(f"x must have shape [M,N] or [B,M,N], got {tuple(x.shape)}")
    if any(size <= 0 for size in x.shape):
        raise ValueError("all input dimensions must be positive")

    squeeze_batch = x.ndim == 2
    work = x.unsqueeze(0) if squeeze_batch else x
    transposed = work.size(-2) > work.size(-1)
    if transposed:
        work = work.mT.contiguous()

    out = torch.ops.muon_ascend950.polar_step(work, float(a), float(b), float(c))
    if transposed:
        out = out.mT.contiguous()
    return out.squeeze(0) if squeeze_batch else out
