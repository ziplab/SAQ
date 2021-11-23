import torch
from torch.autograd import Function


def quantization(x, k):
    n = 2 ** k - 1
    return RoundFunction.apply(x, n)


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
