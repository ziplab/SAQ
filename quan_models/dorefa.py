import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


def quantization(x, k):
    n = 2 ** k - 1
    return RoundFunction.apply(x, n)


def normalization_on_weights(x):
    x = torch.tanh(x)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
    return x


def normalization_on_activations(x, clip_value):
    x = F.relu(x)
    # x = torch.where(x < clip_value, x, clip_value)
    x = torch.clamp(x, max=clip_value)
    return x


def quantize_activation(x, k, clip_value):
    if k == 32:
        return x
    x = normalization_on_activations(x, clip_value)
    x = quantization(x, k)
    return x


def quantize_weight(x, k):
    if k == 32:
        return x
    x = normalization_on_weights(x)
    x = 2 * quantization(x, k) - 1
    return x


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class QConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        bits_weights=32,
        bits_activations=32,
        **kwargs
    ):
        super(QConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        quantized_input = quantize_activation(input, self.bits_activations, 1.0)
        quantized_weight = quantize_weight(self.weight, self.bits_weights)
        output = F.conv2d(
            quantized_input,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("pact_conv")
        return s


class QLinear(nn.Linear):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        bits_weights=32,
        bits_activations=32,
        **kwargs
    ):
        super(QLinear, self).__init__(in_features, out_features, bias=bias)
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        quantized_input = quantize_activation(input, self.bits_activations, 1.0)
        quantized_weight = quantize_weight(self.weight, self.bits_weights)
        output = F.linear(quantized_input, quantized_weight, self.bias)
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("pact_linear")
        return s
