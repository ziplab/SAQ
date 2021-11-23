import torch

from core.warmup_scheduler import GradualWarmupScheduler


def get_optimizer(model, args):
    params = model.parameters()
    if len(args.no_decay_keys) != 0:
        params = []
        for name, param in model.named_parameters():
            flag = False
            for key in args.no_decay_keys:
                if key in name:
                    flag = True
                    break
            if flag:
                weight_decay = 0
            else:
                weight_decay = args.weight_decay
            params.append(
                {"params": param, "lr": args.lr, "weight_decay": weight_decay,}
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


def get_scheduler(optimizer, logger, args):
    if "cosine_warmup" in args.lr_scheduler_type:
        logger.info("Cosine Annealing Warmup LR!")
        after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.n_epochs - args.warmup_n_epochs
        )
        scheduler = GradualWarmupScheduler(
            optimizer, 1, args.warmup_n_epochs, after_scheduler
        )
    elif "cosine" in args.lr_scheduler_type:
        logger.info("Cosine Annealing LR!")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.n_epochs, eta_min=args.min_lr
        )
    elif "multi_step_warmup" in args.lr_scheduler_type:
        logger.info("MultiStep LR Warmup!")
        after_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.step, gamma=0.1
        )
        scheduler = GradualWarmupScheduler(
            optimizer, 1, args.warmup_n_epochs, after_scheduler
        )
    elif "multi_step" in args.lr_scheduler_type:
        logger.info("MultiStep LR!")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.step, gamma=0.1
        )
    else:
        raise NotImplemented
    return scheduler
