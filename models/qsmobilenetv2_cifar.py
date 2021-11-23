"""QSAMSMobileNetV2CIFAR in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch.nn as nn
import torch.nn.functional as F

from . import LIQ_wn_qsam
from .tools import get_conv_fc_quan_type

__all__ = [
    "QSAMSMobileNetV2CifarBlock",
    "QSAMSMobileNetV2Cifar",
    "qsamsmobilenetv2_cifar",
]


class QSAMSMobileNetV2CifarBlock(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(
        self,
        in_planes,
        out_planes,
        expansion,
        stride,
        bits_weights=32,
        bits_activations=32,
        conv_type=LIQ_wn_qsam.QConv2d,
        share_clip=False,
        bits_choice=[2, 3, 4, 5, 6, 7, 8],
    ):
        super(QSAMSMobileNetV2CifarBlock, self).__init__()
        self.stride = stride
        self.bits_choice = bits_choice

        planes = expansion * in_planes
        self.conv1 = conv_type(
            in_planes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            share_clip=share_clip,
            bits_weights_list=bits_choice,
            bits_activations_list=bits_choice,
        )
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm2d(planes) for i in bits_choice for j in bits_choice]
        )
        self.conv2 = conv_type(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            share_clip=share_clip,
            bits_weights_list=bits_choice,
            bits_activations_list=bits_choice,
        )
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm2d(planes) for i in bits_choice for j in bits_choice]
        )
        self.conv3 = conv_type(
            planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            share_clip=share_clip,
            bits_weights_list=bits_choice,
            bits_activations_list=bits_choice,
        )
        # self.bn3 = nn.BatchNorm2d(out_planes)
        self.bn3 = nn.ModuleList(
            [nn.BatchNorm2d(out_planes) for i in bits_choice for j in bits_choice]
        )

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                conv_type(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    bits_weights=bits_weights,
                    bits_activations=bits_activations,
                    share_clip=share_clip,
                    bits_weights_list=bits_choice,
                    bits_activations_list=bits_choice,
                ),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(
            self.bn1[
                self.bits_choice.index(self.conv1.current_bit_weights)
                * len(self.bits_choice)
                + self.bits_choice.index(self.conv1.current_bit_activations)
            ](self.conv1(x))
        )
        out = F.relu(
            self.bn2[
                self.bits_choice.index(self.conv2.current_bit_weights)
                * len(self.bits_choice)
                + self.bits_choice.index(self.conv2.current_bit_activations)
            ](self.conv2(out))
        )
        out = self.bn3[
            self.bits_choice.index(self.conv3.current_bit_weights)
            * len(self.bits_choice)
            + self.bits_choice.index(self.conv3.current_bit_activations)
        ](self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class QSAMSMobileNetV2Cifar(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(
        self,
        num_classes=10,
        quantize_first_last=False,
        bits_weights=32,
        bits_activations=32,
        quan_type="LIQ_wn_qsam",
        share_clip=False,
        bits_choice=[2, 3, 4, 5, 6, 7, 8],
    ):
        super(QSAMSMobileNetV2Cifar, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10

        self.bits_choice = bits_choice
        conv_type, _ = get_conv_fc_quan_type(quan_type)
        first_conv_type, fc_type = get_conv_fc_quan_type("LIQ_wn_qsam")
        if quantize_first_last:
            self.conv1 = first_conv_type(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                bits_weights=8,
                bits_activations=32,
                share_clip=share_clip,
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(
            in_planes=32,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.conv2 = conv_type(
            320,
            1280,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            share_clip=share_clip,
            bits_weights_list=bits_choice,
            bits_activations_list=bits_choice,
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm2d(1280) for i in bits_choice for j in bits_choice]
        )
        # nn.BatchNorm2d(1280)
        if quantize_first_last:
            self.linear = fc_type(
                1280, num_classes, bits_weights=8, bits_activations=8,
            )
        else:
            self.linear = nn.Linear(1280, num_classes)

    def _make_layers(
        self,
        in_planes,
        bits_weights=32,
        bits_activations=32,
        conv_type=LIQ_wn_qsam.QConv2d,
        share_clip=False,
        bits_choice=[2, 3, 4, 5, 6, 7, 8],
    ):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(
                    QSAMSMobileNetV2CifarBlock(
                        in_planes,
                        out_planes,
                        expansion,
                        stride,
                        bits_weights=bits_weights,
                        bits_activations=bits_activations,
                        conv_type=conv_type,
                        share_clip=share_clip,
                        bits_choice=bits_choice,
                    )
                )
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(
            self.bn2[
                self.bits_choice.index(self.conv2.current_bit_weights)
                * len(self.bits_choice)
                + self.bits_choice.index(self.conv2.current_bit_activations)
            ](self.conv2(out))
        )
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def qsamsmobilenetv2_cifar(**kwargs):
    """Constructs a QSAMSMobileNetV2CIFAR model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = QSAMSMobileNetV2Cifar(**kwargs)
    return model

