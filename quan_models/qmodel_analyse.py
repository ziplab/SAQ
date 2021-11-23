import numpy as np
import torch.nn as nn
from core import model_transform as mt
from prettytable import PrettyTable

from .tools import *

__all__ = ["QModelAnalyse"]


class QModelAnalyse(object):
    def __init__(self, model, logger):
        self.model = mt.list2sequential(model)
        self.logger = logger
        self.weight_memory_footprint = []
        self.activation_memory_footprint = []
        self.memory_footprint = []
        self.bops = []
        self.weight_shapes = []
        self.layer_names = []
        self.filter_nums = []
        self.bias_shapes = []
        self.input_shapes = []
        self.output_shapes = []
        self.bits_weights = []
        self.bits_activations = []

    def _qconv_hook(self, layer, x, out):
        if isinstance(out, tuple):
            layer.out_shape = out[0].shape
        else:
            layer.out_shape = out.shape
        layer.in_shape = x[0].shape

    def _qconv_compute_bops(self, layer):
        _, _, h, w = layer.out_shape

        bits_weight = 32
        bits_activation = 32
        if hasattr(layer, "current_bit_weights"):
            bits_weight = layer.current_bit_weights
        elif hasattr(layer, "bits_weights"):
            bits_weight = layer.bits_weights

        if hasattr(layer, "current_bit_activations"):
            bits_activation = layer.current_bit_activations
        elif hasattr(layer, "bits_activations"):
            bits_activation = layer.bits_activations

        layer_name = layer.layer_name
        layer_name = layer_name.replace("module.", "")
        if layer_name == "conv" or layer_name == "conv1":
            if bits_weight != 32:
                bits_activation = 8

        bop = compute_bops(
            layer.kernel_size[0],
            layer.in_channels,
            layer.out_channels // layer.groups,
            h,
            w,
            bits_weight,
            bits_activation,
        )

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(layer.out_shape))
        self.filter_nums.append(layer.out_channels)
        self.bits_weights.append(bits_weight)
        self.bits_activations.append(bits_activation)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.bops.append(bop)

    def _qconv_compute_memory_footprint(self, layer):
        n, c, h, w = layer.in_shape

        bits_weight = 32
        bits_activation = 32
        if hasattr(layer, "current_bit_weights"):
            bits_weight = layer.current_bit_weights
        elif hasattr(layer, "bits_weights"):
            bits_weight = layer.bits_weights

        if hasattr(layer, "current_bit_activations"):
            bits_activation = layer.current_bit_activations
        elif hasattr(layer, "bits_activations"):
            bits_activation = layer.bits_activations

        if layer.layer_name == "conv" or layer.layer_name == "conv1":
            if bits_weight != 32:
                bits_activation = 8

        activation_memory_footprint = compute_memory_footprint(
            n, c, h, w, bits_activation
        )
        weight_memory_footprint = compute_memory_footprint(
            layer.weight.shape[0],
            layer.weight.shape[1],
            layer.weight.shape[2],
            layer.weight.shape[3],
            bits_weight,
        )

        memory_footprint = activation_memory_footprint + weight_memory_footprint

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.input_shapes.append(list(layer.in_shape))
        self.output_shapes.append(list(layer.out_shape))
        self.filter_nums.append(layer.out_channels)
        self.bits_weights.append(bits_weight)
        self.bits_activations.append(bits_activation)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.activation_memory_footprint.append(activation_memory_footprint)
        self.weight_memory_footprint.append(weight_memory_footprint)
        self.memory_footprint.append(memory_footprint)

    def _qlinear_hook(self, layer, x, out):
        layer.in_shape = x[0].shape
        layer.out_shape = out.shape

    def _qlinear_compute_bops(self, layer):
        bits_weight = 32
        bits_activation = 32
        if hasattr(layer, "current_bit_weights"):
            bits_weight = layer.current_bit_weights
        elif hasattr(layer, "bits_weights"):
            bits_weight = layer.bits_weights

        if hasattr(layer, "current_bit_activations"):
            bits_activation = layer.current_bit_activations
        elif hasattr(layer, "bits_activations"):
            bits_activation = layer.bits_activations

        bop = compute_bops(
            1, layer.in_features, layer.out_features, 1, 1, bits_weight, bits_activation
        )

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(layer.out_shape))
        self.filter_nums.append(layer.out_shape[0])
        self.bits_weights.append(bits_weight)
        self.bits_activations.append(bits_activation)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.bops.append(bop)

    def _qlinear_compute_memory_footprint(self, layer):
        n, c = layer.in_shape

        bits_weight = 32
        bits_activation = 32
        if hasattr(layer, "current_bit_weights"):
            bits_weight = layer.current_bit_weights
        elif hasattr(layer, "bits_weights"):
            bits_weight = layer.bits_weights

        if hasattr(layer, "current_bit_activations"):
            bits_activation = layer.current_bit_activations
        elif hasattr(layer, "bits_activations"):
            bits_activation = layer.bits_activations

        activation_memory_footprint = compute_memory_footprint(
            n, c, 1, 1, bits_activation
        )
        weight_memory_footprint = compute_memory_footprint(
            layer.weight.shape[0], layer.weight.shape[1], 1, 1, bits_weight
        )
        memory_footprint = activation_memory_footprint + weight_memory_footprint

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.input_shapes.append(list(layer.in_shape))
        self.output_shapes.append(list(layer.out_shape))
        self.filter_nums.append(layer.out_shape[0])
        self.bits_weights.append(bits_weight)
        self.bits_activations.append(bits_activation)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.activation_memory_footprint.append(activation_memory_footprint)
        self.weight_memory_footprint.append(weight_memory_footprint)
        self.memory_footprint.append(memory_footprint)

    def register_hook(self, hook_list):
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._qconv_hook))
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._qlinear_hook))
            layer.layer_name = layer_name

    def reset(self):
        self.weight_memory_footprint = []
        self.activation_memory_footprint = []
        self.memory_footprint = []
        self.bops = []
        self.weight_shapes = []
        self.layer_names = []
        self.filter_nums = []
        self.bias_shapes = []
        self.input_shapes = []
        self.output_shapes = []
        self.bits_weights = []
        self.bits_activations = []

    def compute_network_bops(self):
        self.reset()

        # compute bops
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                self._qconv_compute_bops(layer)
            elif isinstance(layer, nn.Linear):
                self._qlinear_compute_bops(layer)

        bops_np = np.array(self.bops)
        bops_sum = float(bops_np.sum())
        return bops_sum

    def bops_compute(self, x):
        hook_list = []
        self.register_hook(hook_list)

        # run forward for computing BOPs
        self.model.eval()
        self.model(x)

        bops_sum = self.compute_network_bops()
        for hook in hook_list:
            hook.remove()
        return bops_sum

    def bops_compute_logger(self, x):
        hook_list = []
        self.register_hook(hook_list)

        # run forward for computing BOPs
        self.model.eval()
        self.model(x)

        self.compute_network_bops()
        bops_np = np.array(self.bops)
        bops_sum = float(bops_np.sum())
        percentage = bops_np / bops_sum

        output = PrettyTable()
        output.field_names = [
            "Layer",
            "Weight Shape",
            "#Filters",
            "Bias Shape",
            "Output Shape",
            "BOPs",
            "Percentage",
            "BitW",
            "BitA",
        ]

        self.logger.info("------------------------BOPs------------------------\n")
        for i in range(len(self.bops)):
            output.add_row(
                [
                    self.layer_names[i],
                    self.weight_shapes[i],
                    self.filter_nums[i],
                    self.bias_shapes[i],
                    self.output_shapes[i],
                    bops_np[i],
                    percentage[i],
                    self.bits_weights[i],
                    self.bits_activations[i],
                ]
            )
        self.logger.info(output)
        repo_str = "|===>Total BOPs: {:f} MBOPs".format(bops_sum / 1e6)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()

    def compute_network_memory_footprint(self):
        self.reset()

        # compute bops
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                self._qconv_compute_memory_footprint(layer)
            elif isinstance(layer, nn.Linear):
                self._qlinear_compute_memory_footprint(layer)

        weight_footprint = np.array(self.weight_memory_footprint)
        weight_footprint_sum = float(weight_footprint.sum())
        activation_footprint = np.array(self.activation_memory_footprint)
        activation_footprint_sum = float(activation_footprint.sum())
        activation_footprint_max = float(activation_footprint.max())
        total_footprint = weight_footprint_sum + activation_footprint_sum
        total_footprint_max = weight_footprint_sum + activation_footprint_max
        return total_footprint, total_footprint_max

    def memory_footprint_compute(self, x):
        hook_list = []
        self.register_hook(hook_list)

        # run forward for computing BOPs
        self.model.eval()
        self.model(x)

        total_footprint, total_footprint_max = self.compute_network_memory_footprint()
        for hook in hook_list:
            hook.remove()
        return total_footprint, total_footprint_max

    def memory_footprint_compute_logger(self, x):
        hook_list = []
        self.register_hook(hook_list)

        # run forward for computing BOPs
        self.model.eval()
        self.model(x)

        self.compute_network_memory_footprint()
        weight_footprint = np.array(self.weight_memory_footprint)
        weight_footprint_sum = float(weight_footprint.sum())
        activation_footprint = np.array(self.activation_memory_footprint)
        activation_footprint_sum = float(activation_footprint.sum())
        activation_footprint_max = float(activation_footprint.max())
        total_footprint = weight_footprint_sum + activation_footprint_sum
        total_footprint_max = weight_footprint_sum + activation_footprint_max

        output = PrettyTable()
        output.field_names = [
            "Layer",
            "Weight Shape",
            "#Filters",
            "Bias Shape",
            "Input Shape",
            "Weight Footprint",
            "Activation Footprint",
            "BitW",
            "BitA",
        ]

        self.logger.info("------------------------BOPs------------------------\n")
        for i in range(len(self.weight_memory_footprint)):
            output.add_row(
                [
                    self.layer_names[i],
                    self.weight_shapes[i],
                    self.filter_nums[i],
                    self.bias_shapes[i],
                    self.input_shapes[i],
                    weight_footprint[i],
                    activation_footprint[i],
                    self.bits_weights[i],
                    self.bits_activations[i],
                ]
            )
        self.logger.info(output)
        repo_str = "|===>Total Weight Footprint: {:f} KB".format(
            weight_footprint_sum / 8 / 1e3
        )
        self.logger.info(repo_str)
        repo_str = "|===>Total Activation Footprint: {:f} KB".format(
            activation_footprint_sum / 8 / 1e3
        )
        self.logger.info(repo_str)
        repo_str = "|===>Total Activation Max Footprint: {:f} KB".format(
            activation_footprint_max / 8 / 1e3
        )
        self.logger.info(repo_str)
        repo_str = "|===>Total Footprint: {:f} KB".format(total_footprint / 8 / 1e3)
        self.logger.info(repo_str)
        repo_str = "|===>Total Footprint Max: {:f} KB".format(
            total_footprint_max / 8 / 1e3
        )
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()
