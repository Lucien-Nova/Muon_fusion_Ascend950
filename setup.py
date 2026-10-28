from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import torch_npu


ROOT = Path(__file__).resolve().parent
CANN_HOME = Path(
    os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
).resolve()
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
ASC_SOURCE = ROOT / "csrc" / "muon_polar_unified_ascend950.asc"
BINDING_SOURCE = ROOT / "csrc" / "muon_polar_step_binding.cpp"


def _require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{description} not found: {path}")
    return path


def _require_cann_900() -> None:
    info = _require_path(
        CANN_HOME / "x86_64-linux" / "ascend_all_cann_install.info",
        "CANN installation metadata",
    )
    fields = dict(
        line.split("=", 1)
        for line in info.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    if fields.get("version") != "9.0.0":
        raise RuntimeError(
            f"muon-ascend950 requires CANN 9.0.0, got {fields.get('version')!r} "
            f"from {CANN_HOME}"
        )


class AscendBuildExtension(BuildExtension):
    """Compile the dav-c310 ASC translation unit before linking the Torch op."""

    def build_extensions(self) -> None:
        _require_cann_900()
        bisheng = _require_path(CANN_HOME / "bin" / "bisheng", "Bisheng compiler")
        _require_path(ASC_SOURCE, "unified ASC source")

        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        asc_object = build_temp / "muon_polar_unified_mix.o"
        include_dirs = [
            CANN_HOME / "include",
            CANN_HOME / "x86_64-linux" / "include",
            CANN_HOME / "x86_64-linux" / "asc" / "include",
            CANN_HOME / "x86_64-linux" / "asc" / "include" / "interface",
            CANN_HOME / "x86_64-linux" / "asc" / "impl" / "basic_api",
            CANN_HOME / "x86_64-linux" / "ascendc" / "include",
            CANN_HOME
            / "opp"
            / "built-in"
            / "op_impl"
            / "ai_core"
            / "tbe"
            / "impl"
            / "ops_nn"
            / "ascendc"
            / "common",
        ]
        command = [
            str(bisheng),
            "--cce-aicore-arch=dav-c310",
            "-cce-enable-mix",
            "-DASC_DEVKIT_MAJOR=9",
            "-fPIC",
            "-O3",
            "-g",
            "-D__DAV_C310__",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=false",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "--cce-aicore-jump-expand=true",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            "-std=c++17",
            *(f"-I{path}" for path in include_dirs),
            "-c",
            "--asc-aicore-lang",
            str(ASC_SOURCE),
            "-o",
            str(asc_object),
        ]
        extra_flags = shlex.split(os.environ.get("ASC_EXTRA_FLAGS", ""))
        command[4:4] = extra_flags
        self.announce("compiling unified AscendC source", level=2)
        subprocess.run(command, check=True, cwd=ROOT)

        for extension in self.extensions:
            extension.extra_objects = [str(asc_object)]
        super().build_extensions()


cann_include_dirs = [
    CANN_HOME / "include",
    CANN_HOME / "x86_64-linux" / "include",
    CANN_HOME / "x86_64-linux" / "asc" / "include",
    CANN_HOME / "x86_64-linux" / "ascendc" / "include",
]
cann_library_dirs = [
    CANN_HOME / "lib64",
    CANN_HOME / "x86_64-linux" / "lib64",
]
library_dirs = [TORCH_NPU_ROOT / "lib", *cann_library_dirs]
rpaths = [f"-Wl,-rpath,{path}" for path in library_dirs]

extension = CppExtension(
    name="muon_ascend950._C",
    sources=[str(BINDING_SOURCE)],
    include_dirs=[str(TORCH_NPU_ROOT / "include"), *(str(p) for p in cann_include_dirs)],
    library_dirs=[str(path) for path in library_dirs],
    libraries=[
        "torch_npu",
        "ascendcl",
        "runtime",
        "profapi",
        "unified_dlog",
        "mmpa",
        "c_sec",
        "tiling_api",
        "register",
        "platform",
    ],
    extra_compile_args=["-O2", "-std=c++17"],
    extra_link_args=[
        *rpaths,
        "-Wl,--whole-archive",
        "-lascendc_runtime",
        "-Wl,--no-whole-archive",
        "-ldl",
        "-lpthread",
    ],
)

setup(
    name="muon-ascend950",
    version="0.1.1",
    description="Standalone one-step Polar Express Torch operator for Ascend 950PR",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={
        "build_ext": AscendBuildExtension.with_options(no_python_abi_suffix=True)
    },
    zip_safe=False,
)
