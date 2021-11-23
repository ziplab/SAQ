from torch.nn import functional as F

from .LIQ import QConv2d as LIQQConv2d
from .LIQ import quantize_activation, quantize_weight


class QConv2d(LIQQConv2d):
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
            bits_weights,
            bits_activations,
        )
        self.init_state = False

    def forward(self, input):
        # if not self.init_state:
        #     self.init_state = True
        #     self.init_weight_clip_val()
        #     self.init_activation_clip_val(input)
        quantized_input = quantize_activation(
            input, self.bits_activations, self.activation_clip_value.abs()
        )
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        quantized_weight = quantize_weight(
            normalized_weight, self.bits_weights, self.weight_clip_value.abs()
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
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        max_weight_val = normalized_weight.abs().max() * 0.8
        self.weight_clip_value.data.fill_(max_weight_val)
        print("Init weight clip: {}".format(self.weight_clip_value.data))

    def init_activation_clip_val(self, input):
        max_activation_val = input.abs().max() * 0.8
        self.activation_clip_value.data.fill_(max_activation_val)
        print("Init activation clip: {}".format(self.activation_clip_value.data))

    def extra_repr(self):
        s = super().extra_repr()
        s = s.replace("LIQ_conv2d", "LIQ_wn_conv2d")
        return s

