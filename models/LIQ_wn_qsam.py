from quan_models.LIQ import QConv2d as LIQQConv2d
from quan_models.LIQ import QLinear as LIQQLinear
from quan_models.LIQ import normalization_on_weights, quantization, quantize_activation
from torch.nn import functional as F


class QConv2d(LIQQConv2d):
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
        bits_weights=32,
        bits_activations=32,
        **args
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
            bits_weights,
            bits_activations,
        )
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
            input, self.bits_activations, self.activation_clip_value.abs()
        )
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        if not self.is_second:
            quantized_weight = self.quantize_weight(
                normalized_weight, self.bits_weights, self.weight_clip_value.abs()
            )
        else:
            quantized_weight = self.quantize_weight_add_epsilon(
                normalized_weight,
                self.bits_weights,
                self.weight_clip_value.abs(),
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
        s = s.replace("LIQ_conv2d", "LIQ_wn_qsam_conv2d")
        return s


class QLinear(LIQQLinear):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self, in_features, out_features, bias=True, bits_weights=32, bits_activations=32
    ):
        super(QLinear, self).__init__(
            in_features,
            out_features,
            bias=bias,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
        )
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
        if not self.init_state:
            self.init_state = True
            self.init_weight_clip_val()
            self.init_activation_clip_val(input)
        quantized_input = quantize_activation(
            input, self.bits_activations, self.activation_clip_value.abs()
        )
        if not self.is_second:
            quantized_weight = self.quantize_weight(
                self.weight, self.bits_weights, self.weight_clip_value.abs()
            )
        else:
            quantized_weight = self.quantize_weight_add_epsilon(
                self.weight,
                self.bits_weights,
                self.weight_clip_value.abs(),
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
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("LIQ_qsam_linear")
        return s
