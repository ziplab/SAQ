import torch.nn as nn

from core.engine import get_lr
from core.utils import *
from models import LIQ_wn_qsam
from utils.bypass_bn import disable_running_stats, enable_running_stats


def set_first_forward(model):
    for n, m in model.named_modules():
        if isinstance(m, (LIQ_wn_qsam.QConv2d, LIQ_wn_qsam.QLinear,),):
            m.set_first_forward()


def set_layer_first_forward(model, layer_name):
    for n, m in model.named_modules():
        if (
            isinstance(m, (LIQ_wn_qsam.QConv2d, LIQ_wn_qsam.QLinear,),)
            and n in layer_name
        ):
            m.set_first_forward()


def set_second_forward(model):
    for n, m in model.named_modules():
        if isinstance(m, (LIQ_wn_qsam.QConv2d, LIQ_wn_qsam.QLinear,),):
            m.set_second_forward()


def set_layer_second_forward(model, layer_name):
    for n, m in model.named_modules():
        if (
            isinstance(m, (LIQ_wn_qsam.QConv2d, LIQ_wn_qsam.QLinear,),)
            and n in layer_name
        ):
            m.set_second_forward()


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    minimizer,
    scheduler,
    device,
    logger,
    tensorboard_logger,
    epoch,
    args,
):
    """
        Train one epoch for auxnet
        :param epoch: index of epoch
        """

    metric_logger = MetricLogger(logger=logger, delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    model.train()

    header = "Epoch: [{}]".format(epoch)
    for image, target in metric_logger.log_every(
        train_loader, args.print_frequency, header
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        # Ascent Step
        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True
        # enable_running_stats(model)
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        minimizer.ascent_step()

        # descent step
        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False
        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_second_forward(model)
        # disable_running_stats(model)
        criterion(model(image), target).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        minimizer.descent_step()
        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_first_forward(model)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(
            batch_size * args.world_size / (time.time() - start_time)
        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    scheduler.step()
    lr = get_lr(optimizer)
    logger.info("Change Learning rate: {}".format(lr))

    train_error = 100 - metric_logger.acc1.global_avg
    train_loss = metric_logger.loss.global_avg
    train5_error = 100 - metric_logger.acc5.global_avg
    if tensorboard_logger is not None:
        tensorboard_logger.add_scalar("train_top1_error", train_error, epoch)
        tensorboard_logger.add_scalar("train_top5_error", train5_error, epoch)
        tensorboard_logger.add_scalar("train_loss", train_loss, epoch)
        tensorboard_logger.add_scalar("lr", lr, epoch)
        weight_eps_names = [
            "epsilon",
            "tw_epsilon_norm",
            "normalized_tw_epsilon_norm",
            "weight_clip_value_epsilon",
            "weight_clip_value_tw_epsilon_norm",
            "weight_clip_value_normalized_tw_epsilon_norm",
            "activation_clip_value_epsilon",
            "activation_clip_value_tw_epsilon_norm",
            "activation_clip_value_normalized_tw_epsilon_norm",
            "bias_epsilon",
            "bias_epsilon_norm",
            "bias_normalized_epsilon_norm",
        ]
        bn_eps_names = [
            "weight_epsilon",
            "weight_epsilon_norm",
            "weight_normalized_epsilon_norm",
            "bias_epsilon",
            "bias_epsilon_norm",
            "bias_normalized_epsilon_norm",
        ]
        for name, module in model.named_modules():
            if isinstance(module, (args.conv_type, args.fc_type)):
                if hasattr(module, "weight_clip_value"):
                    tensorboard_logger.add_scalar(
                        "{}_{}".format(name, "weight_clip_value"),
                        module.weight_clip_value,
                        epoch,
                    )
                if hasattr(module, "activation_clip_value"):
                    tensorboard_logger.add_scalar(
                        "{}_{}".format(name, "activation_clip_value"),
                        module.activation_clip_value,
                        epoch,
                    )

                for weight_eps_name in weight_eps_names:
                    if hasattr(module, weight_eps_name):
                        eps = getattr(module, weight_eps_name)
                        if eps.numel() == 1:
                            tensorboard_logger.add_scalar(
                                "{}_{}".format(name, weight_eps_name), eps, epoch,
                            )
                        else:
                            tensorboard_logger.add_histogram(
                                "{}_{}".format(name, weight_eps_name), eps, epoch,
                            )
            elif isinstance(module, (nn.BatchNorm2d)):
                for bn_eps_name in bn_eps_names:
                    if hasattr(module, bn_eps_name):
                        eps = getattr(module, bn_eps_name)
                        if eps.numel() == 1:
                            tensorboard_logger.add_scalar(
                                "{}_{}".format(name, weight_eps_name), eps, epoch,
                            )
                        else:
                            tensorboard_logger.add_histogram(
                                "{}_{}".format(name, weight_eps_name), eps, epoch,
                            )
                        tensorboard_logger.add_histogram(
                            "{}_{}".format(name, bn_eps_name), eps, epoch,
                        )

    logger.info(
        "|===>Training Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(
            train_error, train_loss, train5_error
        )
    )
    return train_error, train_loss, train5_error

