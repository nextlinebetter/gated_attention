import sys
import torch
import pytest

def test_cuda():
    print("python =", sys.executable)
    print("torch  =", torch.__version__)
    print("cuda available =", torch.cuda.is_available())
    assert torch.cuda.is_available()
