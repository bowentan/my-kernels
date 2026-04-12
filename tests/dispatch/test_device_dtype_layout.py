import pytest
import torch
from torch.testing import assert_close

import my_kernels


def test_mismatched_shape():
    a = torch.randn(8, dtype=torch.float32)
    b = torch.randn(10, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same shape"):
        my_kernels.add(a, b)


def test_mismatched_dtype():
    a = torch.randn(8, dtype=torch.float32)
    b = torch.randn(8, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="same data type"):
        my_kernels.add(a, b)


def test_integer_dtype():
    a = torch.ones(10, dtype=torch.int32)
    b = torch.ones(10, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="float32 or float64"):
        my_kernels.add(a, b)


def test_non_contiguous():
    a = torch.randn(8, dtype=torch.float32)[::2]
    b = torch.randn(8, dtype=torch.float32)[::2]

    assert not a.is_contiguous()
    assert not b.is_contiguous()

    out = my_kernels.add(a, b)
    assert_close(out, a + b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_device():
    a = torch.randn(8, dtype=torch.float32, device="cpu")
    b = torch.randn(8, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="same device"):
        my_kernels.add(a, b)
