import os

from core.config import create_dir, get_parser, params_check


def get_qparser():
    parser = get_parser()
    # general
    parser.add_argument(
        "--quantize_first_last",
        type=bool,
        default=True,
        help="whether to quantize the first and last layer",
    )
    parser.add_argument(
        "--quan_type", type=str, default="LIQ_wn", help="type of quantization function"
    )
    parser.add_argument(
        "--clip_lr", type=float, default=0.01, help="learning rate of clip value"
    )
    parser.add_argument("--qw", type=float, default=8.0, help="weight bit")
    parser.add_argument("--qa", type=float, default=8.0, help="activation bit")
    parser.add_argument("--rho", type=float, default=0.1, help="rho in SAM")
    parser.add_argument("--eta", type=float, default=0.01, help="eta in ASAM")
    parser.add_argument(
        "--include_wclip",
        type=bool,
        default=False,
        help="whether to include clip of weight in SAM",
    )
    parser.add_argument(
        "--include_aclip",
        type=bool,
        default=True,
        help="whether to include clip of activation in SAM",
    )
    parser.add_argument(
        "--include_bn",
        type=bool,
        default=True,
        help="whether to include bn parameters in SAM",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5.0, help="maximum norm of gradient",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden dimension of controller"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.5, help="ratio of training set"
    )
    parser.add_argument(
        "--val_num", type=int, default=10000, help="number of validation samples"
    )

    # controller
    parser.add_argument(
        "--c_lr", type=float, default=0.001, help="initial learning rate of controller",
    )
    parser.add_argument(
        "--c_n_epochs",
        type=int,
        default=90,
        help="number of training epochs of controller",
    )
    parser.add_argument(
        "--c_weight_decay", type=float, default=5e-4, help="weight decay of controller",
    )
    parser.add_argument(
        "--c_pretrained", type=str, default=None, help="path of pretrained controller",
    )
    parser.add_argument(
        "--c_resume", type=str, default=None, help="path of resume controller",
    )
    parser.add_argument(
        "--loss_lambda", type=float, default=1e-4, help="lambda coefficient"
    )
    parser.add_argument(
        "--entropy_coeff", type=float, default=5e-4, help="coefficient of entropy term"
    )
    parser.add_argument("--target_bops", type=float, default=648, help="target bops")
    parser.add_argument(
        "--share_clip", type=bool, default=False, help="whether to share clip value"
    )
    parser.add_argument(
        "--bit_warmup_epochs",
        type=int,
        default=0,
        help="number of epochs to warmup sam",
    )
    parser.add_argument(
        "--arch_bits",
        type=lambda s: [float(item) for item in s.split(",")] if len(s) != 0 else "",
        default="",
        help="bits configuration of each layer",
    )
    parser.add_argument(
        "--bits_choice",
        type=lambda s: [float(item) for item in s.split(",")] if len(s) != 0 else "",
        default="2,3,4,5",
        help="bits configuration of each layer",
    )
    parser.add_argument(
        "--wa_same_bit",
        type=bool,
        default=True,
        help="whether to set the same bit to weights and activations",
    )
    parser.add_argument(
        "--search_w_bit",
        type=bool,
        default=False,
        help="whether to set the same bit to weights and activations",
    )
    return parser


def set_save_path(args):
    if len(args.suffix) == 0:
        suffix = "log_{}_{}_bs{:d}_e{:d}_lr{:.5f}_cliplr{}_{}_w{}a{}_qfl{}_opt{}_rho{}_eta{}_iwc{}_iac{}_ib{}_lambda{}_target{}_{}_{}/".format(
            args.network,
            args.dataset,
            args.batch_size,
            args.n_epochs,
            args.lr,
            args.clip_lr,
            args.quan_type,
            args.qw,
            args.qa,
            1 if args.quantize_first_last else 0,
            args.opt_type,
            args.rho,
            args.eta,
            args.include_wclip,
            args.include_aclip,
            args.include_bn,
            args.loss_lambda,
            args.target_bops,
            args.date,
            args.experiment_id,
        )
    else:
        suffix = args.suffix

    args.save_path = os.path.join(args.save_path, suffix)

    create_dir(args.save_path)


def get_args():
    parser = get_qparser()
    args = parser.parse_args()
    params_check(args)
    return args
