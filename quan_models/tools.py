import torch.nn as nn

from . import LIQ, LIQ_wn, dorefa, pact


def get_conv_fc_quan_type(quan_type):
    if quan_type == "LIQ":
        conv_type = LIQ.QConv2d
        fc_type = LIQ.QLinear
    elif quan_type == "LIQ_wn":
        conv_type = LIQ_wn.QConv2d
        fc_type = LIQ.QLinear
    elif quan_type == "pact":
        conv_type = pact.QConv2d
        fc_type = pact.QLinear
    elif quan_type == "dorefa":
        conv_type = dorefa.QConv2d
        fc_type = dorefa.QLinear
    else:
        conv_type = nn.Conv2d
        fc_type = nn.Linear
    return conv_type, fc_type


def compute_bops(
    kernel_size, in_channels, filter_per_channel, h, w, bits_w=32, bits_a=32
):
    conv_per_position_flops = (
        kernel_size * kernel_size * in_channels * filter_per_channel
    )
    active_elements_count = h * w
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bops = overall_conv_flops * bits_w * bits_a
    return bops


def compute_memory_footprint(n, c, h, w, bitwidth=32):
    return n * c * h * w * bitwidth
