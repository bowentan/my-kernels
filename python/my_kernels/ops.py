import torch

from ._load_lib import load_lib


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    load_lib()
    return torch.ops.my_kernels.vector_add(a, b)
