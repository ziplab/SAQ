import torch
import torch.nn as nn
from torch.nn import functional as F

from .common import quantization


def normalization_on_weights(x, clip_value):
    x = x / clip_value
    x = torch.clamp(x, min=-1, max=1)
    return x


def normalization_on_activations(x, clip_value):
    x = x / clip_value
    x = torch.clamp(x, min=0, max=1)
    return x


def quantize_activation(x, k, clip_value):
    if k == 32:
        return x
    x = normalization_on_activations(x, clip_value)
    x = quantization(x, k)
    x = x * clip_value
    return x


def quantize_weight(x, k, clip_value):
    if k == 32:
        return x
    x = normalization_on_weights(x, clip_value)
    x = (x + 1.0) / 2.0
    x = quantization(x, k)
    x = x * 2.0 - 1.0
    x = x * clip_value
    return x


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
        # self.eps = 1e-5
        self.init_state = False
        self.weight_clip_value = nn.Parameter(torch.Tensor([1.0]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([1.0]))
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        if not self.init_state:
            self.init_state = True
            self.init_weight_clip_val()
            self.init_activation_clip_val(input)
        quantized_input = quantize_activation(
            input, self.bits_activations, self.activation_clip_value.abs()
        )
        quantized_weight = quantize_weight(
            self.weight, self.bits_weights, self.weight_clip_value.abs()
        )
        output = F.conv2d(
            quantized_input,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        self.output_shape = output.shape
        return output

    def init_weight_clip_val(self):
        max_weight_val = self.weight.abs().max() * 0.8
        self.weight_clip_value.data.fill_(max_weight_val)
        print("Init weight clip: {}".format(self.weight_clip_value.data))

    def init_activation_clip_val(self, input):
        max_activation_val = input.abs().max() * 0.8
        self.activation_clip_value.data.fill_(max_activation_val)
        print("Init activation clip: {}".format(self.activation_clip_value.data))

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("LIQ_conv2d")
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
        # self.eps = 1e-5
        self.init_state = False
        self.weight_clip_value = nn.Parameter(torch.Tensor([2.0]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([2.0]))
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        if not self.init_state:
            self.init_state = True
            self.init_weight_clip_val()
            self.init_activation_clip_val(input)
        quantized_input = quantize_activation(
            input, self.bits_activations, self.activation_clip_value.abs()
        )
        quantized_weight = quantize_weight(
            self.weight, self.bits_weights, self.weight_clip_value.abs()
        )
        output = F.linear(quantized_input, quantized_weight, self.bias)
        self.output_shape = output.shape
        return output

    def init_weight_clip_val(self):
        max_weight_val = self.weight.abs().max() * 0.8
        self.weight_clip_value.data.fill_(max_weight_val)
        print("FC Init weight clip: {}".format(self.weight_clip_value.data))

    def init_activation_clip_val(self, input):
        max_activation_val = input.abs().max() * 0.8
        self.activation_clip_value.data.fill_(max_activation_val)
        print("FC Init activation clip: {}".format(self.activation_clip_value.data))

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("LIQ_linear")
        return s
