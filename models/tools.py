import torch.nn as nn
from quan_models.LIQ import quantize_activation

from . import LIQ_wn_qsam, switchable_LIQ_wn_qsam


def get_conv_fc_quan_type(quan_type):
    if quan_type == "LIQ_wn_qsam":
        conv_type = LIQ_wn_qsam.QConv2d
        fc_type = LIQ_wn_qsam.QLinear
    elif quan_type == "switchable_LIQ_wn_qsam":
        conv_type = switchable_LIQ_wn_qsam.SwitchableQConv2d
        fc_type = switchable_LIQ_wn_qsam.SwitchableQLinear
    else:
        conv_type = nn.Conv2d
        fc_type = nn.Linear
    return conv_type, fc_type
