# Muon_fusion_Ascend950

针对 **Ascend 950 的 Muon 算子优化**，使用 AscendC 实现 Muon 优化器中 Polar Express 矩阵正交化的单轮计算，提供独立的 PyTorch 接口，可在其他训练项目中调用。

当前编译目标为 **Ascend 950PR（dav-c310）**。Python 包名为 `muon_ascend950`，公开接口为 `polar_step(x, a, b, c)`。五轮迭代、输入归一化和优化器的参数更新由调用方完成。

## 计算与优化方式

每轮计算如下，转置作用于最后两个维度：

```text
A = X @ Xᵀ
P = b × A + c × (A @ A)
Y = a × X + P @ X
```

底层在同一 NPU stream 上顺序启动三个 kernel：

| 阶段 | 计算 | 实现方式 |
| --- | --- | --- |
| Cube1 | `A = X @ Xᵀ` | 只计算上三角，通过 AIV 镜像写回对称矩阵；LF/LF2 路径复用输入 panel |
| Cube2 + P | `P = bA + cA²` | persistent 调度，Cube 结果通过 dual Fixpipe 送入 AIV，在 UB 中融合多项式运算 |
| Cube3 + Y | `Y = aX + PX` | 矩阵乘与 `aX` 加法融合，按输入形状选择 panel 复用策略 |

Cube2/Cube3 无额外 GM workspace；运行时仍会分配 Gram 矩阵 `A`、多项式矩阵 `P` 和输出张量。binding 缓存 tiling 与启动配置，减少重复配置开销。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 设备 | Ascend 950PR，编译架构 `dav-c310` |
| 系统 | Linux；当前构建脚本使用 CANN 的 `x86_64-linux` 目录布局 |
| CANN | **9.0.0**，构建时检查安装元数据中的版本号 |
| PyTorch / torch_npu | 按原工程环境使用匹配的 **2.10 / 2.10** |
| Python | 3.10 或更高版本 |
| 构建与测试依赖 | `setuptools>=68`、`wheel`；运行测试还需 `pytest` |

先按设备环境配置 CANN、驱动、PyTorch 和 torch_npu。扩展依赖构建时的库及链接路径，应在目标环境重新编译。

## 使用流程

以下命令在同一个 Bash 会话中执行。

### 1. 下载并检查环境

```bash
git clone https://github.com/Lucien-Nova/Muon_fusion_Ascend950.git
cd Muon_fusion_Ascend950

export PYTHON_BIN=python3
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0

"$PYTHON_BIN" -c 'import torch, torch_npu; print("torch:", torch.__version__); print("torch_npu:", torch_npu.__version__); print("NPU available:", torch.npu.is_available())'
"$PYTHON_BIN" -m pip install 'setuptools>=68' wheel pytest
```

如果 Python 或 CANN 安装在其他位置，修改上面的两个环境变量。构建会调用 `$ASCEND_HOME_PATH/bin/bisheng`。

### 2. 在源码目录编译

```bash
"$PYTHON_BIN" setup.py build_ext --inplace
```

这会在 `muon_ascend950/` 下生成可加载的 `_C.so`，便于直接运行仓库示例和测试。源码树中已有的嵌套文件 `muon_ascend950/muon_ascend950/_C.so` 不在默认加载位置，不能替代这一步。

### 3. 运行五轮示例

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" examples/five_steps.py
```

这里通过 `PYTHONPATH` 让示例加载刚刚就地编译的源码包。示例使用 `npu:0` 上的 BF16 输入，先归一化，再连续调用五次 `polar_step`。成功运行时的预期输出为：

```text
torch.Size([8, 512, 2048])
```

这表示输出形状与输入一致；精度检查见后文测试流程。

### 4. 构建 wheel 并安装到其他项目

```bash
bash scripts/build_wheel.sh
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps dist/muon_ascend950-*.whl
```

构建脚本不会自动安装 torch、torch_npu 或构建依赖，请先完成环境准备。安装后可在其他项目目录直接导入 `muon_ascend950`。如果只安装 wheel 而未进行源码就地编译，请从仓库外运行 Python，避免当前目录的源码包遮蔽已安装的包。

## Python 调用

下面的五轮系数与仓库示例保持一致：

```python
import torch
import torch_npu  # 注册 NPU 后端
from muon_ascend950 import polar_step

torch.npu.set_device(0)

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

print(x.shape)
```

调用一次 `polar_step` 完成一轮，不会在内部执行归一化或五轮循环。

### 接口约束

| 项目 | 行为 |
| --- | --- |
| 输入设备与类型 | NPU 上的 `torch.bfloat16` 张量 |
| 输入形状 | `[M, N]` 或 `[B, M, N]`，所有维度为正数 |
| 系数 | `a`、`b`、`c` 为标量，包装层转换为 Python `float` |
| `M > N` | Python 包装层先转置，计算后转置回来 |
| 非对齐输入 | 转置策略处理后，M 向上补齐到 128 的倍数；N 补齐到 16 的倍数且不小于补齐后的 M，计算后裁剪 |
| 输出 | 返回新张量，形状和 BF16 类型与输入一致 |

导入包时自动加载 native library，并注册 FakeTensor 实现，供 `torch.compile` 接入使用；编译模式需要在目标环境单独验证，现有测试未覆盖该路径。此接口提供正交化计算，不包含完整 Muon 优化器的动量、权重衰减和参数更新。

## 结果展示与验证

当前仓库包含源码、示例和精度测试，**尚未收录 Ascend 950 的实测耗时、加速比或测试运行日志**。上面的形状输出属于示例预期输出；下面列出可复现的精度检查与性能测量流程。

### 精度测试

在仓库根目录执行，脚本会先就地编译，再运行 pytest：

```bash
ASCEND_DEVICE_ID=0 bash scripts/test.sh
```

现有 `tests/test_polar_step.py` 定义了 4 个测试案例：

| 场景 | 输入形状 | 检查内容 |
| --- | --- | --- |
| 二维、非对齐输入 | `[129, 130]` | 单轮结果与 PyTorch 参考实现一致，输出形状一致 |
| 三维、非对齐输入 | `[2, 150, 2050]` | 单轮结果与 PyTorch 参考实现一致，输出形状一致 |
| `M > N` | `[1, 130, 129]` | 转置路径的单轮结果与参考实现一致，输出形状一致 |
| 外部五轮循环 | `[1, 129, 130]` | 五轮结果与 PyTorch 参考实现一致 |

精度判断使用 `torch.testing.assert_close`，将结果转为 FP32 后比较，容差为 `rtol=6e-2`、`atol=6e-2`。这是测试判定标准，并非已测得的误差。当前测试没有单独覆盖完全对齐输入。

### 性能对比复现

完成源码就地编译后，在仓库根目录运行以下代码，对比**单轮**融合算子与 NPU 上的 PyTorch eager 参考实现。计时包含 Python 调用和设备执行开销，使用同步隔离预热与测量区间；不包含构建、输入生成和归一化。

```bash
unset MUON_POLAR_DEBUG_SYNC
"$PYTHON_BIN" - <<'PY'
import statistics
import time

import torch
import torch_npu
from muon_ascend950 import polar_step

torch.npu.set_device(0)
a, b, c = 8.156554524902461, -22.48329292557795, 15.878769915207462

def reference(x):
    gram = x @ x.mT
    return a * x + (b * gram + c * (gram @ gram)) @ x

def fused(x):
    return polar_step(x, a, b, c)

def measure_us(fn, x, warmup=20, iterations=100, repeats=5):
    for _ in range(warmup):
        fn(x)
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn(x)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return statistics.median(samples)

torch.manual_seed(0)
print("shape | torch eager (us) | fused (us) | speedup")
with torch.no_grad():
    for shape in [(8, 512, 2048), (129, 130), (2, 150, 2050)]:
        x = torch.randn(shape, device="npu:0", dtype=torch.bfloat16)
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        torch.testing.assert_close(
            fused(x).float(), reference(x).float(), rtol=6e-2, atol=6e-2
        )
        baseline_us = measure_us(reference, x)
        fused_us = measure_us(fused, x)
        print(f"{shape} | {baseline_us:.2f} | {fused_us:.2f} | {baseline_us / fused_us:.2f}x")
PY
```

`speedup = PyTorch eager 耗时 / 融合算子耗时`，大于 1 表示当前配置下融合实现更快。发布结果时请同时记录设备型号、CANN/PyTorch/torch_npu 版本、输入形状、调优变量和计时方法。该对比衡量单轮算子，完整五轮和端到端训练吞吐需要另行测量。

## 调优与排查

默认配置按输入形状选择部分 panel 参数，建议先使用默认值验证精度，再逐项调整。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MUON_POLAR_CUBE1_LOWLEVEL_DIM` | `28` | Cube1 核数配置，取值范围 1–128 |
| `MUON_POLAR_CUBE2_PERSISTENT_DIM` | `40` | Cube2 persistent 核数配置，取值范围 1–40 |
| `MUON_POLAR_CUBE2_PANEL_M` | 根据形状自动选择 | Cube2 的 M 方向 panel 复用参数 |
| `MUON_POLAR_CUBE3_DIM` | `40` | Cube3 核数配置，取值范围 1–40 |
| `MUON_POLAR_CUBE3_B_FULL_LOAD` | 根据形状自动选择 | 是否启用 Cube3 的 B panel full-load 路径 |
| `MUON_POLAR_CUBE3_PANEL_M` | 根据形状自动选择 | Cube3 full-load 路径的 panel 复用参数 |
| `MUON_POLAR_DEBUG_SYNC` | 未设置 | 设置后在三个阶段分别同步，用于定位异步执行错误 |

性能测量前使用 `unset MUON_POLAR_DEBUG_SYNC`；源码按变量是否存在决定是否同步，设置为 `0` 也会启用同步。

如果提示 native library 缺失，重新运行 `"$PYTHON_BIN" setup.py build_ext --inplace`；如果更换了 torch、torch_npu 或 CANN 环境，也应重新构建。特殊部署可通过 `MUON_ASCEND950_LIBRARY` 指定兼容的 `.so` 文件路径。

## 工程结构

```text
Muon_fusion_Ascend950/
├── csrc/
│   ├── muon_polar_step_binding.cpp        # PyTorch 注册、输入补齐、tiling 与启动
│   └── muon_polar_unified_ascend950.asc   # 三阶段 AscendC 实现
├── muon_ascend950/                       # Python 接口与 native library 加载
├── examples/five_steps.py                # 五轮 Polar Express 示例
├── tests/test_polar_step.py              # 精度与形状检查
├── scripts/build_wheel.sh                # wheel 构建
├── scripts/test.sh                       # 就地编译并运行测试
├── setup.py
├── pyproject.toml
└── LICENSE
```

## 来源与许可证

工程最初提取自 [SwanHubX/nanochat-ascend PR #15](https://github.com/SwanHubX/nanochat-ascend/pull/15) 的 `muon_ascend950/` 目录，原始提交为 `ae3d3a6ba9bd6fdd9a304b3ac1c1055d822eeaa3`。代码采用 [MIT License](LICENSE)。
