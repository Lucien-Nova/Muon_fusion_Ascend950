from __future__ import annotations

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from muon_ascend950 import polar_step


COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _reference(x: torch.Tensor, coefficients) -> torch.Tensor:
    out = x
    for a, b, c in coefficients:
        gram = out @ out.mT
        polynomial = b * gram + c * (gram @ gram)
        out = a * out + polynomial @ out
    return out


@pytest.fixture(scope="module", autouse=True)
def _select_device():
    torch.npu.set_device(int(os.environ.get("ASCEND_DEVICE_ID", "0")))


@pytest.mark.parametrize("shape", [(129, 130), (2, 150, 2050), (1, 130, 129)])
def test_one_step_matches_reference(shape):
    x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    expected = _reference(x, COEFFICIENTS[:1])
    actual = polar_step(x, *COEFFICIENTS[0])
    torch.testing.assert_close(actual.float(), expected.float(), rtol=6e-2, atol=6e-2)
    assert actual.shape == x.shape


def test_external_five_step_loop():
    x = torch.randn((1, 129, 130), device="npu", dtype=torch.bfloat16)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    expected = _reference(x, COEFFICIENTS)
    actual = x
    for coefficients in COEFFICIENTS:
        actual = polar_step(actual, *coefficients)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=6e-2, atol=6e-2)

