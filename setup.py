from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent


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
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
