# muon-ascend950

面向 Ascend 950PR 的独立 PyTorch Polar Express 单轮算子仓，不依赖 Nanochat。
设备实现由一个自包含的 `muon_polar_unified_ascend950.asc` 编译，Python 只暴露
一个接口：

```python
from muon_ascend950 import polar_step

y = polar_step(x, a, b, c)
```

每次调用完成一轮：

```text
A = X @ X.T
P = b*A + c*(A@A)
Y = a*X + P@X
```

完整五轮由调用方直接循环，不在包内重复封装。

## 环境

- Ascend 950PR（dav-c310）
- CANN `/usr/local/Ascend/cann-9.0.0`
- PyTorch 2.10 + torch_npu 2.10
- Python 3.10+

wheel 包含与构建环境绑定的 C++/AscendC 二进制，因此需要在目标
PyTorch、torch_npu 和 CANN 环境中构建。

## 构建 wheel

```bash
cd csrc/nanochat_ascend/muon_ascend950
PYTHON_BIN=/path/to/python \
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0 \
bash scripts/build_wheel.sh
```

安装生成的 wheel：

```bash
python -m pip install dist/muon_ascend950-*.whl
```

## 跨项目调用

```python
import torch
import torch_npu  # noqa: F401
from muon_ascend950 import polar_step

coefficients = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

x = torch.randn(8, 512, 2048, device="npu:0", dtype=torch.bfloat16)
x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
for a, b, c in coefficients:
    x = polar_step(x, a, b, c)
```

接口已注册 FakeTensor 实现，可直接放入 `torch.compile(fullgraph=True)` 区域；
导入 `muon_ascend950` 时会自动加载 native library。

## 输入约束

- 输入类型：`torch.bfloat16`。
- 输入设备：NPU。
- 输入形状：`[M,N]` 或 `[B,M,N]`。
- Python 接口自动处理 `M > N` 的转置。
- binding 将 M 补齐到 128 的倍数，将 N 补齐到 16 的倍数，并在一轮结束后
  裁剪回原始形状；已对齐输入不产生 padding/crop。
- 输入归一化由调用方在五轮循环前完成，算子内部不会重复归一化。
- Cube2/Cube3 不使用 GM workspace；0.1.1 起已删除无效的 workspace
  参数、分配和缓存字段。

## 测试

```bash
ASCEND_DEVICE_ID=7 \
PYTHON_BIN=/path/to/python \
bash scripts/test.sh
```

测试覆盖对齐、非对齐、二维、三维、`M > N` 和外部五轮循环。

## 调优环境变量

正式实现只有 LF/LF2 Cube1、persistent Cube2+P、low-level Cube3+Y。以下变量
只调整最终路径的 core/panel 参数：

```text
MUON_POLAR_CUBE1_LOWLEVEL_DIM=28
MUON_POLAR_CUBE2_PERSISTENT_DIM=40
MUON_POLAR_CUBE2_PANEL_M=<auto>
MUON_POLAR_CUBE3_DIM=40
MUON_POLAR_CUBE3_B_FULL_LOAD=<auto>
MUON_POLAR_CUBE3_PANEL_M=<auto>
MUON_POLAR_DEBUG_SYNC=1
```
