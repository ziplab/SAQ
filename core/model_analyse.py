import numpy as np
import torch.nn as nn
from prettytable import PrettyTable

from . import model_transform as mt

__all__ = ["ModelAnalyse"]


class ModelAnalyse(object):
    def __init__(self, model, logger):
        self.model = mt.list2sequential(model)
        self.logger = logger
        self.madds = []
        self.weight_shapes = []
        self.layer_names = []
        self.filter_nums = []
        self.channel_nums = []
        self.bias_shapes = []
        self.output_shapes = []

    def _madds_conv_hook(self, layer, x, out):
        input = x[0]
        batch_size = input.shape[0]
        output_height, output_width = out.shape[2:]

        kernel_height, kernel_width = layer.kernel_size
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        groups = layer.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = (
            kernel_height * kernel_width * in_channels * filters_per_channel
        )

        active_elements_count = batch_size * output_height * output_width

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops
        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        self.channel_nums.append(in_channels)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.madds.append(overall_flops)

    def _madds_linear_hook(self, layer, x, out):
        # compute number of multiply-add
        # layer_madds = layer.weight.size(0) * layer.weight.size(1)
        # if layer.bias is not None:
        #     layer_madds += layer.weight.size(0)
        input = x[0]
        batch_size = input.shape[0]
        overall_flops = int(batch_size * input.shape[1] * out.shape[1])

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out.shape[1]
        overall_flops = overall_flops + bias_flops
        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.channel_nums.append(input.shape[1])
        self.output_shapes.append(list(out.shape))
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.madds.append(overall_flops)

    def params_count(self):
        params_num_list = []

        output = PrettyTable()
        output.field_names = ["Param name", "Shape", "Dim"]

        self.logger.info(
            "------------------------number of parameters------------------------\n"
        )
        for name, param in self.model.named_parameters():
            param_num = param.numel()
            param_shape = [shape for shape in param.shape]
            params_num_list.append(param_num)
            output.add_row([name, param_shape, param_num])
        self.logger.info(output)

        params_num_list = np.array(params_num_list)
        params_num = params_num_list.sum()
        self.logger.info(
            "|===>Number of parameters is: {:}, {:f} M".format(
                params_num, params_num / 1e6
            )
        )
        return params_num

    def madds_compute(self, x):
        """
        Compute number of multiply-adds of the model
        """

        hook_list = []
        self.madds = []
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._madds_conv_hook))
                layer.layer_name = layer_name
                # self.layer_names.append(layer_name)
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._madds_linear_hook))
                layer.layer_name = layer_name
                # self.layer_names.append(layer_name)
        # run forward for computing FLOPs
        self.model.eval()
        self.model(x)

        madds_np = np.array(self.madds)
        madds_sum = float(madds_np.sum())
        percentage = madds_np / madds_sum

        output = PrettyTable()
        output.field_names = [
            "Layer",
            "Weight Shape",
            "#Channels",
            "Bias Shape",
            "Output Shape",
            "Madds",
            "Percentage",
        ]

        self.logger.info("------------------------Madds------------------------\n")
        for i in range(len(self.madds)):
            output.add_row(
                [
                    self.layer_names[i],
                    self.weight_shapes[i],
                    self.channel_nums[i],
                    self.bias_shapes[i],
                    self.output_shapes[i],
                    madds_np[i],
                    percentage[i],
                ]
            )
        self.logger.info(output)
        repo_str = "|===>Total MAdds: {:f} M".format(madds_sum / 1e6)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()

        return madds_np
