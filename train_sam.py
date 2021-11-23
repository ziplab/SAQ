import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import core.models as c_models
import models
import quan_models as customized_models
from c_engine import set_bits, set_w_bits, set_wae_bits, show_bits
from core.checkpoint import CheckPoint
from core.config import copy_code
from core.dataloader import get_dataloader, get_random_input
from core.engine import val
from core.label_smooth import LabelSmoothCrossEntropyLoss
from core.logger import get_logger
from core.optim import get_scheduler
from core.utils import *
from core.write_log import write_log, write_settings
from engine import train
from qconfig import get_args, set_save_path
from quan_models.qmodel_analyse import QModelAnalyse
from quan_models.tools import get_conv_fc_quan_type
from utils.optim import get_minimizer

for name in c_models.__dict__:
    if (
        name.islower()
        and not name.startswith("__")
        and callable(c_models.__dict__[name])
    ):
        models.__dict__[name] = c_models.__dict__[name]

for name in customized_models.__dict__:
    if (
        name.islower()
        and not name.startswith("__")
        and callable(customized_models.__dict__[name])
    ):
        models.__dict__[name] = customized_models.__dict__[name]


def get_optimizer(model, args):
    params = []
    for name, param in model.named_parameters():
        if "clip_value" in name:
            lr = args.clip_lr
        else:
            lr = args.lr
        params.append(
            {"params": param, "lr": lr, "weight_decay": args.weight_decay,}
        )
    if "SGDM" in args.opt_type:
        optimizer = torch.optim.SGD(
            params=params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif "SGD" in args.opt_type:
        optimizer = torch.optim.SGD(
            params=params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    elif "RMSProp" in args.opt_type:
        optimizer = torch.optim.RMSprop(
            params=params,
            lr=args.lr,
            alpha=args.alpha,
            eps=args.eps,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    elif "AdamW" in args.opt_type:
        optimizer = torch.optim.AdamW(
            params=params, lr=args.lr, weight_decay=args.weight_decay,
        )
    elif "Adam" in args.opt_type:
        optimizer = torch.optim.Adam(
            params=params, lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        raise NotImplemented
    return optimizer


if __name__ == "__main__":
    # get args
    args = get_args()

    # set gpu
    set_gpu(args)
    device = torch.device("cuda")

    # init distribution
    args.world_size = 1
    init_distributed_mode(args)
    set_save_path(args)
    if is_main_process():
        write_settings(args)
    if args.distributed:
        torch.distributed.barrier()

    # set logger
    logger = get_logger(args.save_path, "main")
    setup_logger_for_distributed(args.rank == 0, logger)

    # set tensorboard logger
    tensorboard_logger = SummaryWriter(args.save_path)
    setup_tensorboard_logger_for_distributed(args.rank == 0, tensorboard_logger)

    # backup code
    copy_code(
        logger, src=os.path.abspath("./"), dst=os.path.join(args.save_path, "code")
    )
    logger.info(args)
    logger.info("|===>Result will be saved at {}".format(args.save_path))

    # get loader and model
    set_reproducible(args.seed)
    train_loader, val_loader, train_sampler, val_sampler = get_dataloader(args, logger)
    random_input = get_random_input(args).to(device)
    model = models.__dict__[args.network](
        num_classes=args.n_classes,
        quantize_first_last=args.quantize_first_last,
        quan_type=args.quan_type,
        bits_weights=args.qw,
        bits_activations=args.qa,
    )
    logger.info(model)

    if len(args.arch_bits) != 0:
        if args.wa_same_bit:
            set_wae_bits(model, args.arch_bits)
        elif args.search_w_bit:
            set_w_bits(model, args.arch_bits)
        else:
            set_bits(model, args.arch_bits)
        show_bits(model)
        logger.info("Set arch bits to: {}".format(args.arch_bits))
        logger.info(model)

    # get checkpoint
    checkpoint = CheckPoint(args.save_path, logger)
    # load pretrained
    if args.pretrained is not None:
        check_point_params = torch.load(args.pretrained, map_location="cpu")
        model_state = check_point_params
        if "model" in check_point_params:
            model_state = check_point_params["model"]
        new_model_state = {}
        for key, value in model_state.items():
            if "module." in key:
                new_key = key.replace("module.", "")
            else:
                new_key = key
            new_model_state[new_key] = value
        model = checkpoint.load_state(model, new_model_state)
        logger.info("|===>load restrain file: {}".format(args.pretrained))

    # load resume
    start_epoch = 0
    optimizer_state = None
    lr_scheduler_state = None
    if args.resume is not None:
        (
            model_state,
            optimizer_state,
            epoch,
            lr_scheduler_state,
        ) = checkpoint.load_checkpoint(args.resume)
        new_model_state = {}
        for key, value in model_state.items():
            if "module." in key:
                new_key = key.replace("module.", "")
            else:
                new_key = key
            new_model_state[new_key] = value
        model = checkpoint.load_state(model, new_model_state)
        start_epoch = epoch + 1
        optimizer_state = optimizer_state
        lr_scheduler_state = lr_scheduler_state
        logger.info("|===>load resume file: {}".format(args.resume))
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.init_state = True

    # move model to gpu
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # get criterion
    if args.label_smooth > 0:
        criterion = LabelSmoothCrossEntropyLoss(num_classes=args.n_classes)
    else:
        criterion = nn.CrossEntropyLoss()
    logger.info("Criterion: {}".format(criterion))

    # get optimizer and scheduler
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, logger, args)
    # get minimizer
    minimizer = get_minimizer(model, optimizer, args)
    logger.info("Optimizer: {}".format(optimizer))
    logger.info("Scheduler: {}".format(scheduler))
    logger.info("Minimizer: {}".format(minimizer))

    if optimizer_state is not None:
        logger.info("Load optimizer state!")
        optimizer.load_state_dict(optimizer_state)
        logger.info("Current lr: {}".format(optimizer.param_groups[0]["lr"]))

    if lr_scheduler_state is not None:
        logger.info("Load lr state")
        scheduler.load_state_dict(lr_scheduler_state)
        logger.info(scheduler.last_epoch)
        logger.info(scheduler.get_last_lr())

    # for module in model.modules():
    #     if isinstance(module, (nn.Conv2d, nn.Linear)):
    #         module.init_state = True

    args.conv_type, args.fc_type = get_conv_fc_quan_type(args.quan_type)
    qmodel_analyse = QModelAnalyse(model, logger)
    qmodel_analyse.bops_compute_logger(random_input)

    if args.resume is None:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.init_state = False

    best_top1 = 100
    best_top5 = 100

    for epoch in range(start_epoch, args.n_epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if epoch < args.bit_warmup_epochs:
            minimizer.rho = 0
        else:
            minimizer.rho = args.rho

        # train for one epoch
        train_error, train_loss, train5_error = train(
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
        )

        # evaluate on validation set
        val_error, val_loss, val5_error = val(
            model,
            val_loader,
            criterion,
            device,
            logger,
            tensorboard_logger,
            epoch,
            args,
        )

        # write log
        log_str = "{:d}\t".format(epoch)
        log_str += "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(
            train_error, train_loss, val_error, val_loss, train5_error, val5_error
        )

        if args.rank == 0:
            write_log(args.save_path, "log.txt", log_str)

        # remember best acc@1, acc@5
        is_best = val_error <= best_top1
        best_top1 = min(best_top1, val_error)
        best_top5 = min(best_top5, val5_error)

        logger.info(
            "|===>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}\n".format(
                best_top1, best_top5
            )
        )
        logger.info(
            "|==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}\n".format(
                100 - best_top1, 100 - best_top5
            )
        )

        # save checkpoint
        if args.rank == 0:
            if "imagenet" in args.dataset:
                checkpoint.save_checkpoint(
                    unwrap_model(model), optimizer, scheduler, epoch, epoch
                )
            else:
                checkpoint.save_checkpoint(
                    unwrap_model(model), optimizer, scheduler, epoch
                )

            if is_best:
                checkpoint.save_model(unwrap_model(model), best_flag=is_best)
