import torch
import torch.nn as nn
from quan_models.LIQ import normalization_on_weights, quantization, quantize_activation
from torch.nn import functional as F


class SwitchableQConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization with sam
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
        bits_weights_list=[2, 3, 4, 5, 6, 7, 8],
        bits_activations_list=[2, 3, 4, 5, 6, 7, 8],
        **kwargs
    ):
        super(SwitchableQConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.share_clip = kwargs["share_clip"]
        self.weight_clip_value = nn.Parameter(
            torch.Tensor([1] if self.share_clip else [1] * len(bits_weights_list))
        )
        self.activation_clip_value = nn.Parameter(
            torch.Tensor([1] if self.share_clip else [1] * len(bits_weights_list))
        )
        self.bits_weights = bits_weights_list
        self.bits_activations = bits_activations_list
        self.current_bit_weights = 4.0
        self.current_bit_activations = 4.0
        self.is_second = False
        self.epsilon = None

    def quantize_weight(self, x, k, clip_value):
        if k == 32:
            return x
        x = normalization_on_weights(x, clip_value)
        x = (x + 1.0) / 2.0
        x = quantization(x, k)
        x = x * 2.0 - 1.0
        x = x * clip_value
        self.x = x
        if self.x.requires_grad:
            self.x.retain_grad()
        return self.x

    def quantize_weight_add_epsilon(self, x, k, clip_value, epsilon):
        if k == 32:
            return x
        x = normalization_on_weights(x, clip_value)
        x = (x + 1.0) / 2.0
        x = quantization(x, k)
        x = x * 2.0 - 1.0
        x = x * clip_value
        self.x = x
        if self.x.requires_grad:
            self.x.retain_grad()
        return self.x + epsilon

    def forward(self, input):
        quantized_input = quantize_activation(
            input,
            self.current_bit_activations,
            self.activation_clip_value.abs()
            if self.share_clip
            else self.activation_clip_value[
                self.bits_activations.index(self.current_bit_activations)
            ].abs(),
        )
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        if not self.is_second:
            quantized_weight = self.quantize_weight(
                normalized_weight,
                self.current_bit_weights,
                self.weight_clip_value.abs()
                if self.share_clip
                else self.weight_clip_value[
                    self.bits_weights.index(self.current_bit_weights)
                ].abs(),
            )
        else:
            quantized_weight = self.quantize_weight_add_epsilon(
                normalized_weight,
                self.current_bit_weights,
                self.weight_clip_value.abs()
                if self.share_clip
                else self.weight_clip_value[
                    self.bits_weights.index(self.current_bit_weights)
                ].abs(),
                self.epsilon,
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

    def set_first_forward(self):
        self.is_second = False

    def set_second_forward(self):
        self.is_second = True

    def extra_repr(self):
        s = super().extra_repr()
        s += ", current bits_weights={}".format(self.current_bit_weights)
        s += ", current bits_activations={}".format(self.current_bit_activations)
        s += ", share clip={}".format(self.share_clip)
        s += ", bits_choices={}".format(self.bits_weights)
        s = s.replace("LIQ_conv2d", "LIQ_wn_qsam_switchable_conv2d")
        return s


class SwitchableQLinear(nn.Linear):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        bits_weights_list=[2, 3, 4, 5, 6, 7, 8],
        bits_activations_list=[2, 3, 4, 5, 6, 7, 8],
        **kwargs
    ):
        super(SwitchableQLinear, self).__init__(in_features, out_features, bias=bias)
        self.share_clip = kwargs["share_clip"]
        self.weight_clip_value = nn.Parameter(
            torch.Tensor([1] if self.share_clip else [1] * len(bits_weights_list))
        )
        self.activation_clip_value = nn.Parameter(
            torch.Tensor([1] if self.share_clip else [1] * len(bits_weights_list))
        )
        self.bits_weights = bits_weights_list
        self.bits_activations = bits_activations_list
        self.current_bit_weights = 4.0
        self.current_bit_activations = 4.0
        self.is_second = False
        self.epsilon = None

    def quantize_weight(self, x, k, clip_value):
        if k == 32:
            return x
        x = normalization_on_weights(x, clip_value)
        x = (x + 1.0) / 2.0
        x = quantization(x, k)
        x = x * 2.0 - 1.0
        x = x * clip_value
        self.x = x
        if self.x.requires_grad:
            self.x.retain_grad()
        return self.x

    def quantize_weight_add_epsilon(self, x, k, clip_value, epsilon):
        if k == 32:
            return x
        x = normalization_on_weights(x, clip_value)
        x = (x + 1.0) / 2.0
        x = quantization(x, k)
        x = x * 2.0 - 1.0
        x = x * clip_value
        self.x = x
        if self.x.requires_grad:
            self.x.retain_grad()
        return self.x + epsilon

    def forward(self, input):
        quantized_input = quantize_activation(
            input,
            self.current_bit_activations,
            self.activation_clip_value.abs()
            if self.share_clip
            else self.activation_clip_value[
                self.bits_activations.index(self.current_bit_activations)
            ].abs(),
        )
        if not self.is_second:
            quantized_weight = self.quantize_weight(
                self.weight,
                self.current_bit_weights,
                self.weight_clip_value.abs()
                if self.share_clip
                else self.weight_clip_value[
                    self.bits_weights.index(self.current_bit_weights)
                ].abs(),
            )
        else:
            quantized_weight = self.quantize_weight_add_epsilon(
                self.weight,
                self.current_bit_weights,
                self.weight_clip_value.abs()
                if self.share_clip
                else self.weight_clip_value[
                    self.bits_weights.index(self.current_bit_weights)
                ].abs(),
                self.epsilon,
            )
        output = F.linear(quantized_input, quantized_weight, self.bias)
        self.output_shape = output.shape
        return output

    def set_first_forward(self):
        self.is_second = False

    def set_second_forward(self):
        self.is_second = True

    def extra_repr(self):
        s = super().extra_repr()
        s += ", current bits_weights={}".format(self.current_bit_weights)
        s += ", current bits_activations={}".format(self.current_bit_activations)
        s += ", share clip={}".format(self.share_clip)
        s += ", bits_choices={}".format(self.bits_weights)
        s += ", method={}".format("LIQ_qsam_switchable_linear")
        return s
