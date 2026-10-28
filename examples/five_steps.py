import torch
import torch_npu  # noqa: F401

from muon_ascend950 import polar_step


COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


device = "npu:0"
x = torch.randn(8, 512, 2048, device=device, dtype=torch.bfloat16)
x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
for a, b, c in COEFFICIENTS:
    x = polar_step(x, a, b, c)
print(x.shape)

