import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import models
import quan_models as customized_models
from c_engine import derive_arch, train
from core.checkpoint import CheckPoint
from core.config import copy_code
from core.dataloader import get_random_input, get_train_val_test_loader
from core.engine import val
from core.label_smooth import LabelSmoothCrossEntropyLoss
from core.logger import get_logger
from core.optim import get_optimizer, get_scheduler
from core.utils import *
from core.write_log import write_log, write_settings
from qconfig import get_args, set_save_path
from quan_models.qmodel_analyse import QModelAnalyse
from quan_models.tools import get_conv_fc_quan_type
from utils.controller import Controller, WABEController, WABEControllerDist
from utils.optim import get_minimizer

for name in customized_models.__dict__:
    if (
        name.islower()
        and not name.startswith("__")
        and callable(customized_models.__dict__[name])
    ):
        models.__dict__[name] = customized_models.__dict__[name]

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
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        val_sampler,
        test_sampler,
    ) = get_train_val_test_loader(args, logger)
    random_input = get_random_input(args).to(device)
    model = models.__dict__[args.network](
        num_classes=args.n_classes,
        quantize_first_last=args.quantize_first_last,
        quan_type=args.quan_type,
        bits_weights=args.qw,
        bits_activations=args.qa,
        share_clip=args.share_clip,
        bits_choice=args.bits_choice,
    )
    logger.info(model)

    # get controller
    n_layers = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and "downsample" not in name:
            n_layers += 1
    if args.wa_same_bit or args.search_w_bit:
        controller = WABEController(
            n_layers=n_layers - 2,
            hidden_size=args.hidden_size,
            device=device,
            bits=args.bits_choice,
        )
    else:
        controller = Controller(
            n_layers=n_layers - 2,
            hidden_size=args.hidden_size,
            device=device,
            bits=args.bits_choice,
        )
    args.n_layers = n_layers
    logger.info(controller)

    # get checkpoint
    checkpoint = CheckPoint(args.save_path, logger)
    c_checkpoint = CheckPoint(os.path.join(args.save_path, "controller"), logger)
    # load pretrained
    if args.pretrained is not None:
        check_point_params = torch.load(args.pretrained, map_location="cpu")
        model_state = check_point_params
        if args.network in ["mobilenetv1"]:
            model_state = check_point_params["model"]
        model = checkpoint.load_state(model, model_state)
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
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.init_state = True
        logger.info("|===>load resume file: {}".format(args.resume))

    if args.c_pretrained is not None:
        check_point_params = torch.load(args.c_pretrained, map_location="cpu")
        controller = c_checkpoint.load_state(controller, check_point_params)
        logger.info("|===>load restrain file: {}".format(args.c_pretrained))

    c_optimizer_state = None
    c_lr_scheduler_state = None
    if args.c_resume is not None:
        (
            controller_state,
            c_optimizer_state,
            epoch,
            c_lr_scheduler_state,
        ) = checkpoint.load_checkpoint(args.c_resume)
        new_model_state = {}
        for key, value in controller_state.items():
            if "module." in key:
                new_key = key.replace("module.", "")
            else:
                new_key = key
            new_model_state[new_key] = value
        controller = checkpoint.load_state(controller, new_model_state)
        c_optimizer_state = c_optimizer_state
        c_lr_scheduler_state = c_lr_scheduler_state
        logger.info("|===>load resume file: {}".format(args.c_resume))

    # move model to gpu
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # move controller to gpu
    controller = controller.to(device)
    # if args.distributed:
    #     controller = torch.nn.parallel.DistributedDataParallel(
    #         controller, device_ids=[args.gpu]
    #     )
    #     controller_without_ddp = controller.module
    logger.info("Controller Bits chocie: {}".format(unwrap_model(controller).bits))

    # get criterion
    if args.label_smooth > 0:
        criterion = LabelSmoothCrossEntropyLoss(num_classes=args.n_classes)
    else:
        criterion = nn.CrossEntropyLoss()
    logger.info("Criterion: {}".format(criterion))

    # get optimizer and scheduler
    optimizer = get_optimizer(model, args)
    controller_optimizer = torch.optim.Adam(
        controller.parameters(),
        args.c_lr,
        betas=(0.5, 0.999),
        weight_decay=args.c_weight_decay,
    )
    scheduler = get_scheduler(optimizer, logger, args)
    # get minimizer
    minimizer = get_minimizer(model, optimizer, args)
    logger.info("Optimizer: {}".format(optimizer))
    logger.info("Controller optimizer: {}".format(controller_optimizer))
    logger.info("Scheduler: {}".format(scheduler))
    logger.info("Minimizer: {}".format(minimizer))
    if optimizer_state is not None:
        logger.info("Load optimizer state!")
        optimizer.load_state_dict(optimizer_state)

    if c_optimizer_state is not None:
        logger.info("Load controller optimizer state!")
        controller_optimizer.load_state_dict(c_optimizer_state)

    if lr_scheduler_state is not None:
        logger.info("Load lr state")
        scheduler.load_state_dict(lr_scheduler_state)
        logger.info(scheduler.last_epoch)
        logger.info(scheduler.get_last_lr())

    args.conv_type, args.fc_type = get_conv_fc_quan_type(args.quan_type)
    qmodel_analyse = QModelAnalyse(model, logger)
    qmodel_analyse.bops_compute_logger(random_input)
    # logger.info(model.conv.is_second)

    val_loader.dataset.transforms = test_loader.dataset.transforms
    logger.info(val_loader.dataset.transforms)
    (
        sharpness_list,
        val_error_list,
        bops_list,
        bits_seq_list,
        entropy_list,
    ) = derive_arch(
        model,
        controller,
        val_loader,
        criterion,
        minimizer,
        device,
        logger,
        qmodel_analyse,
        args,
    )
    min_sharpness = min(sharpness_list)
    min_sharpness_index = sharpness_list.index(min_sharpness)
    min_error = min(val_error_list)
    min_index = val_error_list.index(min_error)
    logger.info("Min index: {}".format(min_index))
    logger.info("Min sharpness index: {}".format(min_sharpness_index))
    logger.info("Sharpness list: {}".format(sharpness_list))
    logger.info("Val error list: {}".format(val_error_list))
    logger.info("Min sharpness: {}".format(min_sharpness))
    logger.info("Min error: {}".format(min_error))

    logger.info("Bits seq for min error: {}".format(bits_seq_list[min_index]))
    if not args.wa_same_bit and not args.search_w_bit:
        logger.info("Weight Bits: {}".format(bits_seq_list[min_index][::2]))
        logger.info("Activation Bits: {}".format(bits_seq_list[min_index][1::2]))
    # logger.info("Weight Bits: {}".format(bits_seq_list[min_index][::2]))
    # logger.info("Activation Bits: {}".format(bits_seq_list[min_index][1::2]))
    logger.info("Entropy: {}".format(entropy_list[min_index]))
    logger.info("BOPs: {}".format(bops_list[min_index]))

    logger.info(
        "Bits seq for min sharpness: {}".format(bits_seq_list[min_sharpness_index])
    )
    if not args.wa_same_bit and not args.search_w_bit:
        logger.info("Weight Bits: {}".format(bits_seq_list[min_sharpness_index][::2]))
        logger.info(
            "Activation Bits: {}".format(bits_seq_list[min_sharpness_index][1::2])
        )
    # logger.info("Weight Bits: {}".format(bits_seq_list[min_index][::2]))
    # logger.info("Activation Bits: {}".format(bits_seq_list[min_index][1::2]))
    logger.info("Entropy: {}".format(entropy_list[min_sharpness_index]))
    logger.info("BOPs: {}".format(bops_list[min_sharpness_index]))
