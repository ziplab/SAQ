import argparse
import os
import shutil
from datetime import datetime

from core.utils import is_main_process


def get_parser():
    parser = argparse.ArgumentParser("classification")

    # general
    parser.add_argument("--save_path", type=str, default="", help="output directory")
    parser.add_argument(
        "--suffix", type=str, default="", help="suffix of save dir name"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--gpu", type=str, default="0", help="GPU id to use, e.g. 0,1,2,3"
    )
    parser.add_argument(
        "--print_frequency", type=int, default=10, help="print frequency"
    )
    parser.add_argument(
        "--dist_url", type=str, default="env://", help="distributed URL"
    )

    # data
    parser.add_argument("--data_path", type=str, default="", help="path of dataset")
    parser.add_argument(
        "--dataset", type=str, default="cifar100", help="imagenet | cifar10 | cifar100",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default="4",
        help="number of threads used for data loading",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.today().strftime("%Y%m%d"),
        help="date of experiment",
    )
    parser.add_argument(
        "--experiment_id", type=str, default="01", help="Id of experiment",
    )
    parser.add_argument(
        "--use_dali_cpu",
        type=bool,
        default=False,
        help="whether to use cpu in data loading",
    )

    # optimization
    parser.add_argument("--batch_size", type=int, default=128, help="mini-batch size")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum term")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="minimum learning rate of cosine scheduler",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--step",
        type=lambda s: [int(item) for item in s.split(",")],
        default="80, 120",
        help="multi-step for linear learning rate",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="multi_step", help="multi_step"
    )
    parser.add_argument("--opt_type", type=str, default="SGD", help="optimizer")
    parser.add_argument(
        "--warmup_n_epochs", type=int, default=0, help="number of warmup epoch"
    )
    parser.add_argument(
        "--label_smooth", type=float, default=0, help="ratio of label smooth"
    )
    parser.add_argument(
        "--no_decay_keys",
        type=lambda s: [item for item in s.split(",")] if len(s) != 0 else "",
        default="",
        help="key name that does not apply weight decay",
    )

    # model
    parser.add_argument(
        "--network", type=str, default="preresnet20", help="network name"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="path of pretrained model"
    )
    parser.add_argument("--resume", type=str, default=None, help="path of resume model")
    return parser


def params_check(args):
    if args.dataset in ["cifar10"]:
        args.n_classes = 10
    elif args.dataset in ["cifar100"]:
        args.n_classes = 100
    elif "imagenet" in args.dataset:
        args.n_classes = 1000


def create_dir(save_path):
    if is_main_process():
        if os.path.exists(save_path):
            print("{} file exist!".format(save_path))
            action = input("Select Action: d (delete) / q (quit):").lower().strip()
            act = action
            if act == "d":
                shutil.rmtree(save_path)
            else:
                raise OSError("Directory {} exits!".format(save_path))

        if not os.path.exists(save_path):
            os.makedirs(save_path)


def set_save_path(args):
    if len(args.suffix) == 0:
        suffix = "log_{}_{}_bs{:d}_e{:d}_lr{:.5f}_step{}_{}_{}/".format(
            args.network,
            args.dataset,
            args.batch_size,
            args.n_epochs,
            args.lr,
            args.step,
            args.date,
            args.experiment_id,
        )
    else:
        suffix = args.suffix
    args.save_path = os.path.join(args.save_path, suffix)

    create_dir(args.save_path)


def copy_code(logger, src=os.path.abspath("./"), dst="./code/"):
    """
        copy code in current path to a folder
        """

    if is_main_process():
        for f in os.listdir(src):
            if "output" in f or "log" in f:
                continue
            src_file = os.path.join(src, f)
            file_split = f.split(".")
            if len(file_split) >= 2 and file_split[1] == "py":
                if not os.path.isdir(dst):
                    os.makedirs(dst)
                dst_file = os.path.join(dst, f)
                try:
                    shutil.copyfile(src=src_file, dst=dst_file)
                except:
                    logger.errro(
                        "copy file error! src: {}, dst: {}".format(src_file, dst_file)
                    )
            elif os.path.isdir(src_file):
                deeper_dst = os.path.join(dst, f)
                copy_code(logger, src=src_file, dst=deeper_dst)


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    params_check(args)
    return args
