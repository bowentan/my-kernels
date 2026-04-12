import pytest
import torch
from torch.testing import assert_close

import my_kernels
from tests.reference.add import add_ref

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.float64]
SHAPES = [
    (0,),
    (1,),
    (1 << 10,),
    (1 << 20,),
    (17,),
    (4, 128),
    (2, 7, 11),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_add(device, dtype, shape):
    torch.manual_seed(0)

    a = torch.randn(*shape, dtype=dtype, device=device)
    b = torch.randn(*shape, dtype=dtype, device=device)

    expected = add_ref(a, b)
    actual = my_kernels.add(a, b)

    assert_close(actual, expected)
