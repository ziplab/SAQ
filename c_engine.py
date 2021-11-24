import torch.nn as nn

from core.engine import get_lr, val
from core.utils import *
from engine import (set_first_forward, set_layer_first_forward,
                    set_layer_second_forward, set_second_forward)
from models.qmobilenetv2_cifar import QSAMMobileNetV2CifarBlock
from models.qpreresnet import QSAMPreBasicBlock
from models.qresnet import QSAMBasicBlock, QSAMBottleneck
from models.qsmobilenetv2_cifar import QSAMSMobileNetV2CifarBlock
from models.qspreresnet import QSAMSPreBasicBlock
from models.qsresnet import QSAMSBasicBlock, QSAMSBottleneck
from utils.controller import Controller


def set_bits(model, bits_seq):
    layer_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d)) and "downsample" not in name:
            layer_idx += 1
            if layer_idx == 1:
                continue
            else:
                weight_bit = bits_seq[(layer_idx - 2) * 2]
                activation_bit = bits_seq[(layer_idx - 2) * 2 + 1]
                m.current_bit_weights = weight_bit
                m.current_bit_activations = activation_bit
    # set bits of downsampling layer to the same bit of conv2
    for name, m in model.named_modules():
        if isinstance(m, (QSAMSPreBasicBlock, QSAMPreBasicBlock)) and m.downsample:
            m.downsample.current_bit_weights = m.conv2.current_bit_weights
            m.downsample.current_bit_activations = m.conv2.current_bit_activations
        elif isinstance(m, (QSAMSBasicBlock, QSAMBasicBlock)) and m.downsample:
            m.downsample[0].current_bit_weights = m.conv2.current_bit_weights
            m.downsample[0].current_bit_activations = m.conv2.current_bit_activations
        elif isinstance(m, (QSAMSBottleneck, QSAMBottleneck)) and m.downsample:
            m.downsample[0].current_bit_weights = m.conv3.current_bit_weights
            m.downsample[0].current_bit_activations = m.conv3.current_bit_activations
        elif (
            isinstance(m, (QSAMMobileNetV2CifarBlock, QSAMSMobileNetV2CifarBlock))
            and m.shortcut
        ):
            m.shortcut[0].current_bit_weights = m.conv3.current_bit_weights
            m.shortcut[0].current_bit_activations = m.conv3.current_bit_activations


def set_wae_bits(model, bits_seq):
    layer_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d)) and "downsample" not in name:
            layer_idx += 1
            if layer_idx == 1:
                continue
            else:
                weight_bit = bits_seq[layer_idx - 2]
                m.current_bit_weights = weight_bit
                m.current_bit_activations = weight_bit
    # set bits of downsampling layer to the same bit of conv2
    for name, m in model.named_modules():
        if isinstance(m, (QSAMSPreBasicBlock, QSAMPreBasicBlock)) and m.downsample:
            m.downsample.current_bit_weights = m.conv2.current_bit_weights
            m.downsample.current_bit_activations = m.conv2.current_bit_activations
        elif isinstance(m, (QSAMSBasicBlock, QSAMBasicBlock)) and m.downsample:
            m.downsample[0].current_bit_weights = m.conv2.current_bit_weights
            m.downsample[0].current_bit_activations = m.conv2.current_bit_activations
        elif isinstance(m, (QSAMSBottleneck, QSAMBottleneck)) and m.downsample:
            m.downsample[0].current_bit_weights = m.conv3.current_bit_weights
            m.downsample[0].current_bit_activations = m.conv3.current_bit_activations
        elif (
            isinstance(m, (QSAMMobileNetV2CifarBlock, QSAMSMobileNetV2CifarBlock))
            and m.shortcut
        ):
            m.shortcut[0].current_bit_weights = m.conv3.current_bit_weights
            m.shortcut[0].current_bit_activations = m.conv3.current_bit_activations


def show_bits(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if hasattr(m, "current_bit_weights"):
                print(
                    "Layer: {}, Bits W: {}, Bits A: {}".format(
                        name, m.current_bit_weights, m.current_bit_activations
                    )
                )
            else:
                print(
                    "Layer: {}, Bits W: {}, Bits A: {}".format(
                        name, m.bits_weights, m.bits_activations
                    )
                )


def set_w_bits(model, bits_seq):
    layer_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d)) and "downsample" not in name:
            layer_idx += 1
            if layer_idx == 1:
                continue
            else:
                weight_bit = bits_seq[layer_idx - 2]
                m.current_bit_weights = weight_bit
                m.current_bit_activations = 4.0
    # set bits of downsampling layer to the same bit of conv2
    for name, m in model.named_modules():
        if isinstance(m, (QSAMSPreBasicBlock, QSAMPreBasicBlock)) and m.downsample:
            m.downsample.current_bit_weights = m.conv2.current_bit_weights
            m.downsample.current_bit_activations = m.conv2.current_bit_activations
        elif isinstance(m, (QSAMSBasicBlock, QSAMBasicBlock)) and m.downsample:
            m.downsample[0].current_bit_weights = m.conv2.current_bit_weights
            m.downsample[0].current_bit_activations = m.conv2.current_bit_activations
        elif isinstance(m, (QSAMSBottleneck, QSAMBottleneck)) and m.downsample:
            m.downsample[0].current_bit_weights = m.conv3.current_bit_weights
            m.downsample[0].current_bit_activations = m.conv3.current_bit_activations
        elif (
            isinstance(m, (QSAMMobileNetV2CifarBlock, QSAMSMobileNetV2CifarBlock))
            and m.shortcut
        ):
            m.shortcut[0].current_bit_weights = m.conv3.current_bit_weights
            m.shortcut[0].current_bit_activations = m.conv3.current_bit_activations


def get_loss(image, target, model, criterion, minimizer, args):
    # Ascent Step
    model.require_backward_grad_sync = False
    model.require_forward_param_sync = True
    output = model(image)
    loss = criterion(output, target)
    loss.backward()
    minimizer.ascent_step()

    # descent step
    model.require_backward_grad_sync = True
    model.require_forward_param_sync = False
    if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
        set_second_forward(model)
    loss = criterion(model(image), target)

    if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
        set_first_forward(model)
    minimizer.restore_step()
    return loss


def get_reward(
    image, target, model, criterion, minimizer, qmodel_analyse, bits_seq, args
):
    if args.wa_same_bit:
        set_wae_bits(model, bits_seq)
    elif args.search_w_bit:
        set_w_bits(model, bits_seq)
    else:
        set_bits(model, bits_seq)
    # model.eval()
    loss = get_loss(image, target, model, criterion, minimizer, args)
    bops = qmodel_analyse.compute_network_bops()
    if "imagenet" in args.dataset:
        computation_loss = (bops / 1e9 - args.target_bops) ** 2
    else:
        computation_loss = (bops / 1e6 - args.target_bops) ** 2
    reward = loss + args.loss_lambda * computation_loss
    return (reward, bops, loss, computation_loss)


def controller_step(
    model,
    controller,
    qmodel_analyse,
    val_iter,
    criterion,
    controller_optimizer,
    minimizer,
    device,
    args,
):
    image, target = next(val_iter)
    image, target = image.to(device), target.to(device)

    bits_seq, probs, logp, entropy = controller.forward()
    reward, bops, loss, computation_loss = get_reward(
        image, target, model, criterion, minimizer, qmodel_analyse, bits_seq, args
    )
    policy_loss = logp * reward
    controller_loss = logp * reward - args.entropy_coeff * entropy

    controller_optimizer.zero_grad()
    policy_loss.backward()
    controller_optimizer.step()
    return (
        controller_loss,
        policy_loss,
        entropy,
        probs,
        logp,
        reward,
        bops,
        loss,
        computation_loss,
        bits_seq,
    )


def controller_train(
    model,
    controller,
    val_loader,
    criterion,
    controller_optimizer,
    minimizer,
    device,
    logger,
    tensorboard_logger,
    qmodel_analyse,
    epoch,
    args,
):
    """
        Train one epoch
        :param epoch: index of epoch
        """

    metric_logger = MetricLogger(logger=logger, delimiter="  ")
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    controller.train()
    model.eval()

    header = "Controller Epoch: [{}]".format(epoch)
    for image, target in metric_logger.log_every(
        val_loader, args.print_frequency, header
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        bits_seq, probs, logp, entropy = controller.forward()
        if is_dist_avail_and_initialized():
            dist.broadcast(logp, src=0)
            dist.broadcast(entropy, src=0)
        reward, bops, loss, computation_loss = get_reward(
            image, target, model, criterion, minimizer, qmodel_analyse, bits_seq, args
        )
        policy_loss = logp * reward
        controller_loss = logp * reward - args.entropy_coeff * entropy

        controller_optimizer.zero_grad()
        controller_loss.backward()
        controller_optimizer.step()

        batch_size = image.shape[0]
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        metric_logger.update(
            controller_loss=controller_loss.item(),
            policy_loss=policy_loss.item(),
            entropy=entropy.item(),
            logp=logp.item(),
            reward=reward.item(),
            bops=(bops / 1e9) if "imagenet" in args.dataset else (bops / 1e6),
            c_ce_loss=loss.item(),
            c_comp_loss=computation_loss,
        )

    if tensorboard_logger is not None:
        tensorboard_logger.add_scalar(
            "policy_loss", metric_logger.policy_loss.global_avg, epoch
        )
        tensorboard_logger.add_scalar(
            "controller_loss", metric_logger.controller_loss.global_avg, epoch
        )
        tensorboard_logger.add_scalar(
            "entropy", metric_logger.entropy.global_avg, epoch
        )
        tensorboard_logger.add_scalar("logp", metric_logger.logp.global_avg, epoch)
        tensorboard_logger.add_scalar("reward", metric_logger.reward.global_avg, epoch)
        tensorboard_logger.add_scalar("bops", metric_logger.bops.global_avg, epoch)
        tensorboard_logger.add_scalar(
            "c_ce_loss", metric_logger.c_ce_loss.global_avg, epoch
        )
        tensorboard_logger.add_scalar(
            "c_comp_loss", metric_logger.c_comp_loss.global_avg, epoch
        )

        layer_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_idx += 1
                if layer_idx == 1 or (layer_idx - 2) * 2 >= len(probs):
                    continue
                if args.wa_same_bit or args.search_w_bit:
                    layer_weight_probs = probs[layer_idx]
                    layer_activation_probs = probs[layer_idx]
                else:
                    layer_weight_probs = probs[(layer_idx - 2) * 2]
                    layer_activation_probs = probs[(layer_idx - 2) * 2 + 1]
                logger.info(layer_weight_probs)
                logger.info(layer_activation_probs)
                for bit_idx, bit in enumerate(args.bits_choice):
                    tensorboard_logger.add_scalar(
                        "{}_weight_bit{}_probs".format(name, bit),
                        layer_weight_probs[0][bit_idx].item(),
                        epoch,
                    )

                    tensorboard_logger.add_scalar(
                        "{}_activation_bit{}_probs".format(name, bit),
                        layer_activation_probs[0][bit_idx].item(),
                        epoch,
                    )
    logger.info("Bits seq: {}".format(bits_seq))
    if not args.wa_same_bit and not args.search_w_bit:
        logger.info("Weight Bits: {}".format(bits_seq[::2]))
        logger.info("Activation Bits: {}".format(bits_seq[1::2]))


def model_train(
    model,
    controller,
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
    metric_logger = MetricLogger(logger=logger, delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    model.train()
    controller.eval()

    header = "Model Epoch: [{}]".format(epoch)
    for image, target in metric_logger.log_every(
        train_loader, args.print_frequency, header
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        # sample arch
        if epoch < args.bit_warmup_epochs:
            bits_seq = unwrap_model(controller).random_sample()
        else:
            bits_seq, probs, logp, entropy = controller.forward()
        if args.wa_same_bit:
            set_wae_bits(model, bits_seq)
        elif args.search_w_bit:
            set_w_bits(model, bits_seq)
        else:
            set_bits(model, bits_seq)

        # Ascent Step
        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        minimizer.ascent_step()

        # descent step
        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False
        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_second_forward(model)
        criterion(model(image), target).backward()
        minimizer.descent_step()
        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_first_forward(model)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

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
                    for wc_idx in range(len(module.weight_clip_value)):
                        tensorboard_logger.add_scalar(
                            "{}_{}_{}".format(name, "weight_clip_value", wc_idx),
                            module.weight_clip_value[wc_idx],
                            epoch,
                        )
                if hasattr(module, "activation_clip_value"):
                    for ac_idx in range(len(module.activation_clip_value)):
                        tensorboard_logger.add_scalar(
                            "{}_{}_{}".format(name, "activation_clip_value", ac_idx),
                            module.activation_clip_value[ac_idx],
                            epoch,
                        )

                for weight_eps_name in weight_eps_names:
                    if (
                        hasattr(module, weight_eps_name)
                        and getattr(module, weight_eps_name) is not None
                    ):
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


def train(
    model,
    controller,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    controller_optimizer,
    minimizer,
    scheduler,
    device,
    logger,
    tensorboard_logger,
    qmodel_analyse,
    epoch,
    args,
):
    """
        Train one epoch
        :param epoch: index of epoch
        """

    metric_logger = MetricLogger(logger=logger, delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    model.train()
    controller.train()

    header = "Epoch: [{}]".format(epoch)
    val_iter = iter(val_loader)
    for image, target in metric_logger.log_every(
        train_loader, args.print_frequency, header
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        # architecture step
        (
            controller_loss,
            policy_loss,
            entropy,
            probs,
            logp,
            reward,
            bops,
            ce_loss,
            computation_loss,
            bits_seq,
        ) = controller_step(
            model,
            controller,
            qmodel_analyse,
            val_iter,
            criterion,
            controller_optimizer,
            minimizer,
            device,
            args,
        )

        # Ascent Step
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        minimizer.ascent_step()

        # descent step
        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_second_forward(model)
        criterion(model(image), target).backward()
        minimizer.descent_step()
        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_first_forward(model)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(
            policy_loss=policy_loss.item(),
            entropy=entropy.item(),
            logp=logp.item(),
            reward=reward.item(),
            bops=bops / 1e6,
            c_ce_loss=ce_loss.item(),
            c_comp_loss=computation_loss,
        )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

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
        tensorboard_logger.add_scalar(
            "policy_loss", metric_logger.policy_loss.global_avg, epoch
        )
        tensorboard_logger.add_scalar(
            "entropy", metric_logger.entropy.global_avg, epoch
        )
        tensorboard_logger.add_scalar("logp", metric_logger.logp.global_avg, epoch)
        tensorboard_logger.add_scalar("reward", metric_logger.reward.global_avg, epoch)
        tensorboard_logger.add_scalar("bops", metric_logger.bops.global_avg, epoch)
        tensorboard_logger.add_scalar(
            "c_ce_loss", metric_logger.c_ce_loss.global_avg, epoch
        )
        tensorboard_logger.add_scalar(
            "c_comp_loss", metric_logger.c_comp_loss.global_avg, epoch
        )

        layer_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_idx += 1
                if layer_idx == 1 or (layer_idx - 2) * 2 >= len(probs):
                    continue
                layer_weight_probs = probs[(layer_idx - 2) * 2]
                layer_activation_probs = probs[(layer_idx - 2) * 2 + 1]
                logger.info(layer_weight_probs)
                logger.info(layer_activation_probs)
                for bit_idx, bit in enumerate(args.bits_choice):
                    tensorboard_logger.add_scalar(
                        "{}_weight_bit{}_probs".format(name, bit),
                        layer_weight_probs[0][bit_idx].item(),
                        epoch,
                    )

                    tensorboard_logger.add_scalar(
                        "{}_activation_bit{}_probs".format(name, bit),
                        layer_activation_probs[0][bit_idx].item(),
                        epoch,
                    )

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
                    for wc_idx in range(len(module.weight_clip_value)):
                        tensorboard_logger.add_scalar(
                            "{}_{}_{}".format(name, "weight_clip_value", wc_idx),
                            module.weight_clip_value[wc_idx],
                            epoch,
                        )
                if hasattr(module, "activation_clip_value"):
                    for ac_idx in range(len(module.activation_clip_value)):
                        tensorboard_logger.add_scalar(
                            "{}_{}_{}".format(name, "activation_clip_value", ac_idx),
                            module.activation_clip_value[ac_idx],
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
    logger.info("Bits seq: {}".format(bits_seq))
    logger.info("Weight Bits: {}".format(bits_seq[::2]))
    logger.info("Activation Bits: {}".format(bits_seq[1::2]))
    return train_error, train_loss, train5_error


def compute_sharpness(
    model, train_loader, criterion, minimizer, device, logger, args,
):
    metric_logger = MetricLogger(logger=logger, delimiter="  ")

    model.eval()

    header = "Epoch: [{}]".format(0)
    # accumulate gradient for all data
    for image, target in metric_logger.log_every(
        train_loader, args.print_frequency, header
    ):
        image, target = image.to(device), target.to(device)

        # Ascent Step
        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        metric_logger.update(loss=loss.item())

    if args.rho == 0:
        sharpness = metric_logger.loss.global_avg
    else:
        minimizer.ascent_step()
        # descent step
        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False
        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_second_forward(model)

        for image, target in metric_logger.log_every(
            train_loader, args.print_frequency, header
        ):
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)
            metric_logger.update(loss_w_epsilon=loss.item())
        sharpness = (
            metric_logger.loss_w_epsilon.global_avg - metric_logger.loss.global_avg
        )
        minimizer.restore_step()

        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_first_forward(model)
    return sharpness


def compute_layer_weight_sharpness(
    model, train_loader, criterion, minimizer, device, logger, args,
):

    model.eval()

    sharpness_list = []
    name_list = []
    for module_n, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        # for _ in ["weight", "activation"]:
        for _ in ["weight"]:
            logger.info(
                "Processing layer: {}, weight/activation: {}".format(module_n, _)
            )
            metric_logger = MetricLogger(logger=logger, delimiter="  ")

            param_name_list = []
            for param_n, param in module.named_parameters():
                if _ in param_n:
                    param_name_list.append("{}.{}".format(module_n, param_n))

            logger.info(param_name_list)

            header = "Epoch: [{}]".format(0)
            # accumulate gradient for all data
            for image, target in metric_logger.log_every(
                train_loader, args.print_frequency, header
            ):
                image, target = image.to(device), target.to(device)

                # Ascent Step
                output = model(image)
                loss = criterion(output, target)
                loss.backward()
                metric_logger.update(loss=loss.item())

            logger.info("Loss: {}".format(metric_logger.loss.global_avg))

            minimizer.ascent_step_param(param_name_list)

            # descent step
            if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
                set_layer_second_forward(model, module_n)

            metric_logger_loss_w_epsilon = MetricLogger(logger=logger, delimiter="  ")
            for image, target in metric_logger_loss_w_epsilon.log_every(
                train_loader, args.print_frequency, header
            ):
                image, target = image.to(device), target.to(device)

                output = model(image)
                loss = criterion(output, target)
                metric_logger_loss_w_epsilon.update(loss=loss.item())
            sharpness = (
                metric_logger_loss_w_epsilon.loss.global_avg
                - metric_logger.loss.global_avg
            )
            logger.info("Layer: {}, Sharpness: {}".format(module_n, sharpness))
            minimizer.restore_step_param(param_name_list)

            if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
                set_layer_first_forward(model, module_n)
            sharpness_list.append(sharpness)
            name_list.append("{}.{}".format(module_n, _))

    return sharpness_list, name_list


def compute_layer_activation_sharpness(
    model, train_loader, criterion, minimizer, device, logger, args,
):

    model.eval()

    sharpness_list = []
    sharpness_delta_list = []
    name_list = []
    module_name_list_before_this_layer = []
    module_list_before_this_layer = []
    for module_n, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        if len(module_list_before_this_layer) == 0:
            module_name_list_before_this_layer.append(module_n)
            module_list_before_this_layer.append(module)
            continue

        logger.info("Processing layer: {}".format(module_n))
        metric_logger = MetricLogger(logger=logger, delimiter="  ")

        param_name_list = []
        for sub_module_n, sub_module in zip(
            module_name_list_before_this_layer, module_list_before_this_layer
        ):
            for param_n, param in sub_module.named_parameters():
                if "weight" in param_n:
                    param_name_list.append("{}.{}".format(sub_module_n, param_n))

        logger.info(param_name_list)

        header = "Epoch: [{}]".format(0)
        # accumulate gradient for all data
        for image, target in metric_logger.log_every(
            train_loader, args.print_frequency, header
        ):
            image, target = image.to(device), target.to(device)

            # Ascent Step
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            metric_logger.update(loss=loss.item())

        logger.info("Loss: {}".format(metric_logger.loss.global_avg))

        before_step = {}
        for sub_param_n, sub_param, in model.named_parameters():
            before_step[sub_param_n] = sub_param.clone()

        minimizer.ascent_step_param(param_name_list)

        # descent step
        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_layer_second_forward(model, module_name_list_before_this_layer)

        metric_logger_loss_w_epsilon = MetricLogger(logger=logger, delimiter="  ")
        for image, target in metric_logger_loss_w_epsilon.log_every(
            train_loader, args.print_frequency, header
        ):
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)
            metric_logger_loss_w_epsilon.update(loss=loss.item())
        sharpness = (
            metric_logger_loss_w_epsilon.loss.global_avg - metric_logger.loss.global_avg
        )
        sharpness_delta = (
            abs(sharpness - sharpness_list[-1])
            if len(sharpness_delta_list) > 0
            else sharpness
        )
        logger.info("Loss: {}".format(metric_logger_loss_w_epsilon.loss.global_avg))
        logger.info("Layer: {}, Sharpness: {}".format(module_n, sharpness))
        minimizer.restore_step_param(param_name_list)

        after_step = {}
        for sub_param_n, sub_param, in model.named_parameters():
            after_step[sub_param_n] = sub_param.clone()

        for k, v in before_step.items():
            close_num = torch.isclose(before_step[k], after_step[k]).sum()
            if close_num != before_step[k].nelement():
                logger.info("Param {} changed!!!".format(k))
                assert False

        if "QSAM" in args.opt_type or "QASAM" in args.opt_type:
            set_layer_first_forward(model, module_name_list_before_this_layer)
        sharpness_list.append(sharpness)
        sharpness_delta_list.append(sharpness_delta)
        name_list.append("{}.{}".format(module_n, "activation"))
        module_name_list_before_this_layer.append(module_n)
        module_list_before_this_layer.append(module)

    return sharpness_list, sharpness_delta_list, name_list


def derive_arch(
    model,
    controller,
    val_loader,
    criterion,
    minimizer,
    device,
    logger,
    qmodel_analyse,
    args,
):
    i = 0
    sharpness_list = []
    val_error_list = []
    bops_list = []
    bits_seq_list = []
    entropy_list = []
    controller.eval()
    model.eval()
    while i != 20:
        bits_seq, probs, logp, entropy = controller.forward()
        if is_dist_avail_and_initialized():
            dist.broadcast(logp, src=0)
            dist.broadcast(entropy, src=0)
        if args.wa_same_bit:
            set_wae_bits(model, bits_seq)
        elif args.search_w_bit:
            set_w_bits(model, bits_seq)
        else:
            set_bits(model, bits_seq)
        if "imagenet" in args.dataset:
            bops = qmodel_analyse.compute_network_bops() / 1e9
        else:
            bops = qmodel_analyse.compute_network_bops() / 1e6
        logger.info("Generate arch with bops {} and entropy {}".format(bops, entropy))
        if "imagenet" in args.dataset:
            if "mobilenetv2" in args.network:
                if bops > args.target_bops or bops < args.target_bops - 0.1:
                    continue
            else:
                if bops > args.target_bops or bops < args.target_bops - 0.2:
                    continue
        else:
            if bops > args.target_bops or bops < args.target_bops - 10:
                continue
        show_bits(model)
        sharpness = compute_sharpness(
            model, val_loader, criterion, minimizer, device, logger, args,
        )
        val_error, val_loss, val5_error = val(
            model, val_loader, criterion, device, logger, None, 0, args,
        )

        sharpness_list.append(sharpness)
        val_error_list.append(val_error)
        bops_list.append(bops)
        bits_seq_list.append(bits_seq)
        entropy_list.append(entropy)
        i += 1
    return sharpness_list, val_error_list, bops_list, bits_seq_list, entropy_list
