import math

import numpy as np
import torch.nn as nn

from . import LIQ_wn_qsam
from .tools import get_conv_fc_quan_type

__all__ = ["QSAMSMobileNetV2", "QSAMSMobileBottleneck", "qsamsmobilenetv2"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def qconv3x3(
    in_planes,
    out_planes,
    stride=1,
    bits_weights=32,
    bits_activations=32,
    conv_type=LIQ_wn_qsam.QConv2d,
    share_clip=False,
    bits_choice=[2, 3, 4, 5, 6, 7, 8],
):
    "3x3 convolution with padding"
    return conv_type(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
        share_clip=share_clip,
        bits_weights_list=bits_choice,
        bits_activations_list=bits_choice,
    )


def conv1x1(in_planes, out_planes):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


def qconv1x1(
    in_planes,
    out_planes,
    bits_weights=32,
    bits_activations=32,
    conv_type=LIQ_wn_qsam.QConv2d,
    share_clip=False,
    bits_choice=[2, 3, 4, 5, 6, 7, 8],
):
    "1x1 convolution"
    return conv_type(
        in_planes,
        out_planes,
        kernel_size=1,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
        share_clip=share_clip,
        bits_weights_list=bits_choice,
        bits_activations_list=bits_choice,
    )


def dwconv3x3(in_planes, out_planes, stride=1):
    "3x3 depth wise convolution"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=in_planes,
        bias=False,
    )


def qdwconv3x3(
    in_planes,
    out_planes,
    stride=1,
    bits_weights=32,
    bits_activations=32,
    conv_type=LIQ_wn_qsam.QConv2d,
    share_clip=False,
    bits_choice=[2, 3, 4, 5, 6, 7, 8],
):
    "3x3 depth wise convolution"
    return conv_type(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=in_planes,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
        share_clip=share_clip,
        bits_weights_list=bits_choice,
        bits_activations_list=bits_choice,
    )


class QSAMSMobileBottleneck(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        expand=1,
        stride=1,
        bits_weights=32,
        bits_activations=32,
        conv_type=LIQ_wn_qsam.QConv2d,
        share_clip=False,
        bits_choice=[2, 3, 4, 5, 6, 7, 8],
    ):
        super(QSAMSMobileBottleneck, self).__init__()
        self.name = "mobile-bottleneck"
        self.expand = expand
        self.bits_choice = bits_choice

        intermedia_planes = in_planes * expand
        if self.expand != 1:
            self.conv1 = qconv1x1(
                in_planes,
                intermedia_planes,
                bits_weights=bits_weights,
                bits_activations=bits_activations,
                conv_type=conv_type,
                share_clip=share_clip,
                bits_choice=bits_choice,
            )
            # self.bn1 = nn.BatchNorm2d(intermedia_planes)
            self.bn1 = nn.ModuleList(
                [nn.BatchNorm2d(intermedia_planes) for i in bits_choice for j in bits_choice]
            )
            self.relu1 = nn.ReLU6(inplace=True)

        self.conv2 = qdwconv3x3(
            intermedia_planes,
            intermedia_planes,
            stride=stride,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        # self.bn2 = nn.BatchNorm2d(intermedia_planes)
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm2d(intermedia_planes) for i in bits_choice for j in bits_choice]
        )
        self.relu2 = nn.ReLU6(inplace=True)

        self.conv3 = qconv1x1(
            intermedia_planes,
            out_planes,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        # self.bn3 = nn.BatchNorm2d(out_planes)
        self.bn3 = nn.ModuleList(
            [nn.BatchNorm2d(out_planes) for i in bits_choice for j in bits_choice]
        )

        self.shortcut = stride == 1 and in_planes == out_planes
        self.block_index = 0

    def forward(self, x):
        if self.shortcut:
            residual = x

        out = x
        if self.expand != 1:
            out = self.conv1(out)
            # out = self.bn1(out)
            out = self.bn1[
                self.bits_choice.index(self.conv1.current_bit_weights)
                * len(self.bits_choice)
                + self.bits_choice.index(self.conv1.current_bit_activations)
            ](out)
            out = self.relu1(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.bn2[
            self.bits_choice.index(self.conv2.current_bit_weights)
            * len(self.bits_choice)
            + self.bits_choice.index(self.conv2.current_bit_activations)
        ](out)
        out = self.relu2(out)

        out = self.conv3(out)
        # out = self.bn3(out)
        out = self.bn3[
            self.bits_choice.index(self.conv3.current_bit_weights)
            * len(self.bits_choice)
            + self.bits_choice.index(self.conv3.current_bit_activations)
        ](out)

        if self.shortcut:
            out += residual

        return out


class QSAMSMobileNetV2(nn.Module):
    """
    MobileNet_v2
    """

    def __init__(
        self,
        num_classes=1000,
        wide_scale=1.0,
        bits_weights=32,
        bits_activations=32,
        quantize_first_last=False,
        quan_type="LIQ_wn_qsam",
        share_clip=False,
        bits_choice=[2, 3, 4, 5, 6, 7, 8],
    ):
        super(QSAMSMobileNetV2, self).__init__()

        block = QSAMSMobileBottleneck
        # define network structure
        self.layer_width = np.array([32, 16, 24, 32, 64, 96, 160, 320])
        self.layer_width = np.around(self.layer_width * wide_scale)
        self.layer_width = self.layer_width.astype(int)

        self.in_planes = self.layer_width[0].item()

        conv_type, _ = get_conv_fc_quan_type(quan_type)
        first_conv_type, fc_type = get_conv_fc_quan_type("LIQ_wn_qsam")
        if quantize_first_last:
            self.conv1 = qconv3x3(
                3,
                self.in_planes,
                stride=2,
                bits_weights=8,
                bits_activations=32,
                conv_type=first_conv_type,
            )
        else:
            self.conv1 = conv3x3(3, self.in_planes, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU6(inplace=True)
        self.layer1 = self._make_layer(
            block,
            self.layer_width[1].item(),
            blocks=1,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.layer2 = self._make_layer(
            block,
            self.layer_width[2].item(),
            blocks=2,
            expand=6,
            stride=2,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.layer3 = self._make_layer(
            block,
            self.layer_width[3].item(),
            blocks=3,
            expand=6,
            stride=2,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.layer4 = self._make_layer(
            block,
            self.layer_width[4].item(),
            blocks=4,
            expand=6,
            stride=2,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.layer5 = self._make_layer(
            block,
            self.layer_width[5].item(),
            blocks=3,
            expand=6,
            stride=1,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.layer6 = self._make_layer(
            block,
            self.layer_width[6].item(),
            blocks=3,
            expand=6,
            stride=2,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.layer7 = self._make_layer(
            block,
            self.layer_width[7].item(),
            blocks=1,
            expand=6,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.conv2 = qconv1x1(
            in_planes=self.layer_width[7].item(),
            out_planes=1280,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.relu2 = nn.ReLU6(inplace=True)
        # self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0)

        if quantize_first_last:
            self.fc = fc_type(1280, num_classes, bits_weights=8, bits_activations=8)
        else:
            self.fc = nn.Linear(1280, num_classes)
        # self.conv3 = conv1x1(1280, num_classes)
        # self.relu3 = nn.ReLU6(inplace=True)
        # self.fc = nn.Linear(self.layer_width[8], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block,
        out_planes,
        blocks,
        expand=1,
        stride=1,
        bits_weights=32,
        bits_activations=32,
        conv_type=LIQ_wn_qsam.QConv2d,
        share_clip=False,
        bits_choice=[2, 3, 4, 5, 6, 7, 8],
    ):
        layers = []
        layers.append(
            block(
                self.in_planes,
                out_planes,
                expand=expand,
                stride=stride,
                bits_weights=bits_weights,
                bits_activations=bits_activations,
                conv_type=conv_type,
                share_clip=share_clip,
                bits_choice=bits_choice,
            )
        )
        self.in_planes = out_planes
        for i in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    out_planes,
                    expand=expand,
                    bits_weights=bits_weights,
                    bits_activations=bits_activations,
                    conv_type=conv_type,
                    share_clip=share_clip,
                    bits_choice=bits_choice,
                )
            )
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward propagation
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.avgpool(x)
        # # x = self.conv3(x)
        x = x.mean([2, 3])
        x = self.dropout(x)
        x = self.fc(x)

        return x


def qsamsmobilenetv2(**kwargs):
    """Constructs a QSAMSMobileNetv2 model.
    """
    model = QSAMSMobileNetV2(**kwargs)
    return model
