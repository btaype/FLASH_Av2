from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parent
OFFICIAL_ROOT = ROOT / "csrc" / "flash_attn_official"
OFFICIAL_SRC = OFFICIAL_ROOT / "src"
OFFICIAL_SOURCES = [
    OFFICIAL_ROOT / "flash_api.cpp",
    *sorted(OFFICIAL_SRC.glob("flash_fwd_hdim*.cu")),
    *sorted(OFFICIAL_SRC.glob("flash_fwd_split_hdim*.cu")),
    *sorted(OFFICIAL_SRC.glob("flash_bwd_hdim*.cu")),
]

OFFICIAL_NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]


setup(
    name="f_attencion_v2",
    version="0.1.0",
    description="Implementacion simple de FlashAttention-2",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="f_attencion_v2_cuda",
            sources=[
                str(ROOT / "csrc" / "binding.cpp"),
                str(ROOT / "csrc" / "flash_fwd.cu"),
                str(ROOT / "csrc" / "flash_bwd.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17"],
                "nvcc": ["-O2", "--use_fast_math", "-std=c++17"],
            },
        ),
        CUDAExtension(
            name="f_attencion_v2_official_cuda",
            sources=[str(path) for path in OFFICIAL_SOURCES],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": OFFICIAL_NVCC_FLAGS,
            },
            include_dirs=[
                str(OFFICIAL_ROOT),
                str(OFFICIAL_SRC),
                str(REPO_ROOT / "csrc" / "cutlass" / "include"),
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
