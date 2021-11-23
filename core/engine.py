from core.utils import *


def get_lr(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group["lr"]
        break
    return lr


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    logger,
    tensorboard_logger,
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

    header = "Epoch: [{}]".format(epoch)
    for image, target in metric_logger.log_every(
        train_loader, args.print_frequency, header
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        # forward
        output = model(image)
        loss = criterion(output, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(
            args.world_size * batch_size / (time.time() - start_time)
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

    logger.info(
        "|===>Training Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(
            train_error, train_loss, train5_error
        )
    )
    return train_error, train_loss, train5_error


def val(model, val_loader, criterion, device, logger, tensorboard_logger, epoch, args):
    """
    Validation
    :param epoch: index of epoch
    """

    model.eval()
    metric_logger = MetricLogger(logger=logger, delimiter="  ")
    header = "Test:"

    with torch.no_grad():
        for image, target in metric_logger.log_every(
            val_loader, args.print_frequency, header
        ):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # forward
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    val_error = 100 - metric_logger.acc1.global_avg
    val_loss = metric_logger.loss.global_avg
    val5_error = 100 - metric_logger.acc5.global_avg
    if tensorboard_logger is not None:
        tensorboard_logger.add_scalar("val_top1_error", val_error, epoch)
        tensorboard_logger.add_scalar("val_top5_error", val5_error, epoch)
        tensorboard_logger.add_scalar("val_loss", val_loss, epoch)

    logger.info(
        "|===>Testing Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(
            val_error, val_loss, val5_error
        )
    )
    return val_error, val_loss, val5_error
