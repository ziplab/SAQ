import math

import torch.nn as nn

from . import LIQ_wn_qsam
from .tools import get_conv_fc_quan_type

__all__ = [
    "QSAMSPreResNet",
    "QSAMSPreBasicBlock",
    "qsamspreresnet20",
    "qsamspreresnet32",
    "qsamspreresnet44",
    "qsamspreresnet56",
    "qsamspreresnet110",
]


# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------


def conv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    """
    return nn.Conv2d(
        in_plane, out_plane, kernel_size=1, stride=stride, padding=0, bias=False
    )


def qconv1x1(
    in_plane,
    out_plane,
    stride=1,
    bits_weights=32,
    bits_activations=32,
    conv_type=LIQ_wn_qsam.QConv2d,
    share_clip=False,
    bits_choice=[2, 3, 4, 5, 6, 7, 8],
):
    """
    1x1 quantized convolutional layer
    """
    return conv_type(
        in_plane,
        out_plane,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
        share_clip=share_clip,
        bits_weights_list=bits_choice,
        bits_activations_list=bits_choice,
    )


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
    "3x3 quantized convolution with padding"
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


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


# both-preact | half-preact


class QSAMSPreBasicBlock(nn.Module):
    """
    base module for PreResNet on small data sets
    """

    def __init__(
        self,
        in_plane,
        out_plane,
        stride=1,
        downsample=None,
        block_type="both_preact",
        bits_weights=32,
        bits_activations=32,
        conv_type=LIQ_wn_qsam.QConv2d,
        share_clip=False,
        bits_choice=[2, 3, 4, 5, 6, 7, 8],
    ):
        """
        init module and weights
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param stride: stride of convolutional layers, default 1
        :param downsample: down sample type for expand dimension of input feature maps, default None
        :param block_type: type of blocks, decide position of short cut, both-preact: short cut start from beginning
        of the first segment, half-preact: short cut start from the position between the first segment and the second
        one. default: both-preact
        """
        super(QSAMSPreBasicBlock, self).__init__()
        self.name = block_type
        self.downsample = downsample
        self.bits_choice = bits_choice

        # self.bn1 = nn.BatchNorm2d(in_plane)
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm2d(in_plane) for i in bits_choice for j in bits_choice]
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = qconv3x3(
            in_plane,
            out_plane,
            stride,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        # self.bn2 = nn.BatchNorm2d(out_plane)
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm2d(out_plane) for i in bits_choice for j in bits_choice]
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = qconv3x3(
            out_plane,
            out_plane,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.block_index = 0

    def forward(self, x):
        """
        forward procedure of residual module
        :param x: input feature maps
        :return: output feature maps
        """
        x, previous_wbit, previous_abit = x
        if self.name == "half_preact":
            x = self.bn1[
                self.bits_choice.index(previous_wbit) * len(self.bits_choice)
                + self.bits_choice.index(previous_abit)
            ](x)
            x = self.relu1(x)
            residual = x
            x = self.conv1(x)
            x = self.bn2[
                self.bits_choice.index(self.conv1.current_bit_weights)
                * len(self.bits_choice)
                + self.bits_choice.index(self.conv1.current_bit_activations)
            ](x)
            x = self.relu2(x)
            x = self.conv2(x)
        elif self.name == "both_preact":
            residual = x
            x = self.bn1[
                self.bits_choice.index(previous_wbit) * len(self.bits_choice)
                + self.bits_choice.index(previous_abit)
            ](x)
            x = self.relu1(x)
            x = self.conv1(x)
            x = self.bn2[
                self.bits_choice.index(self.conv1.current_bit_weights)
                * len(self.bits_choice)
                + self.bits_choice.index(self.conv1.current_bit_activations)
            ](x)
            x = self.relu2(x)
            x = self.conv2(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = x + residual
        return [out, self.conv2.current_bit_weights, self.conv2.current_bit_activations]


class QSAMSPreResNet(nn.Module):
    """
    define QSAMSPreResNet on small data sets
    """

    def __init__(
        self,
        depth,
        quantize_first_last=False,
        wide_factor=1,
        num_classes=10,
        bits_weights=32,
        bits_activations=32,
        quan_type="LIQ_wn_qsam",
        share_clip=False,
        bits_choice=[2, 3, 4, 5, 6, 7, 8],
    ):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        """
        super(QSAMSPreResNet, self).__init__()

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6

        conv_type, _ = get_conv_fc_quan_type(quan_type)
        first_conv_type, fc_type = get_conv_fc_quan_type("LIQ_wn_qsam")

        if quantize_first_last:
            self.conv = qconv3x3(
                3,
                16 * wide_factor,
                bits_weights=8,
                bits_activations=32,
                conv_type=first_conv_type,
            )
        else:
            self.conv = conv3x3(3, 16 * wide_factor)
        self.layer1 = self._make_layer(
            QSAMSPreBasicBlock,
            16 * wide_factor,
            n,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.layer2 = self._make_layer(
            QSAMSPreBasicBlock,
            32 * wide_factor,
            n,
            stride=2,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.layer3 = self._make_layer(
            QSAMSPreBasicBlock,
            64 * wide_factor,
            n,
            stride=2,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
            share_clip=share_clip,
            bits_choice=bits_choice,
        )
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        if quantize_first_last:
            self.fc = fc_type(
                64 * wide_factor, num_classes, bits_weights=8, bits_activations=8
            )
        else:
            self.fc = linear(64 * wide_factor, num_classes)

        self._init_weight()

    def _init_weight(self):
        # init layer parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def _make_layer(
        self,
        block,
        out_plane,
        n_blocks,
        stride=1,
        bits_weights=32,
        bits_activations=32,
        conv_type=LIQ_wn_qsam.QConv2d,
        share_clip=False,
        bits_choice=[2, 3, 4, 5, 6, 7, 8],
    ):
        """
        make residual blocks, including short cut and residual function
        :param block: type of basic block to build network
        :param out_plane: size of output plane
        :param n_blocks: number of blocks on every segment
        :param stride: stride of convolutional neural network, default 1
        :return: residual blocks
        """
        downsample = None
        if stride != 1 or self.in_plane != out_plane:
            downsample = qconv1x1(
                self.in_plane,
                out_plane,
                stride=stride,
                bits_weights=bits_weights,
                bits_activations=bits_activations,
                conv_type=conv_type,
                share_clip=share_clip,
                bits_choice=bits_choice,
            )

        layers = []
        layers.append(
            block(
                self.in_plane,
                out_plane,
                stride,
                downsample,
                block_type="half_preact",
                bits_weights=bits_weights,
                bits_activations=bits_activations,
                conv_type=conv_type,
                share_clip=share_clip,
                bits_choice=bits_choice,
            )
        )
        self.in_plane = out_plane
        for i in range(1, int(n_blocks)):
            layers.append(
                block(
                    self.in_plane,
                    out_plane,
                    bits_weights=bits_weights,
                    bits_activations=bits_activations,
                    conv_type=conv_type,
                    share_clip=share_clip,
                    bits_choice=bits_choice,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps
        """
        out = self.conv(x)
        out = self.layer1([out, 4, 4])
        out = self.layer2(out)
        out = self.layer3(out)
        out, bitw, bita = out
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def qsamspreresnet20(**kwargs):
    """Constructs a QSAMSPreResNet-20 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = QSAMSPreResNet(depth=20, **kwargs)
    return model


def qsamspreresnet32(**kwargs):
    """Constructs a QSAMSPreResNet-32 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = QSAMSPreResNet(depth=32, **kwargs)
    return model


def qsamspreresnet44(**kwargs):
    """Constructs a QSAMSPreResNet-44 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = QSAMSPreResNet(depth=44, **kwargs)
    return model


def qsamspreresnet56(**kwargs):
    """Constructs a QSAMSPreResNet-56 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = QSAMSPreResNet(depth=56, **kwargs)
    return model


def qsamspreresnet110(**kwargs):
    """Constructs a QSAMSPreResNet-110 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = QSAMSPreResNet(depth=110, **kwargs)
    return model
