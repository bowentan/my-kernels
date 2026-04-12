from pathlib import Path

_LOADED = False


def load_lib() -> None:
    global _LOADED
    if _LOADED:
        return

    # Try pre-built extension first (installed via pip install)
    try:
        from . import _C  # noqa: F401

        _LOADED = True
        return
    except ImportError:
        pass

    # Fall back to JIT compilation (dev mode)
    import torch
    from torch.utils.cpp_extension import load, CUDA_HOME

    root = Path(__file__).resolve().parents[2]

    srcs = [
        root / "src" / "pytorch" / "register_ops.cpp",
        root / "src" / "ops" / "add" / "add_cpu.cpp",
    ]

    build_with_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuflags = ["-O3"]
    if build_with_cuda:
        srcs.append(root / "src" / "ops" / "add" / "add_cuda.cu")
        extra_cflags.append("-DWITH_CUDA")

    load(
        name="_C",
        sources=[str(p) for p in srcs],
        extra_include_paths=[str(root / "include")],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuflags,
        with_cuda=build_with_cuda,
        verbose=False,
    )

    _LOADED = True
