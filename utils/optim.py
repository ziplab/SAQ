# import core.asam as asam
import utils.asam as asam
import utils.qasam as qasam
# import utils.qasam as qasam
import utils.qsam as qsam
import utils.sam as sam


def get_minimizer(model, optimizer, args):
    if "QSAM" in args.opt_type:
        minimizer = qsam.QSAM(
            optimizer,
            model,
            rho=args.rho,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "QASAM" in args.opt_type:
        minimizer = qasam.QASAM(
            optimizer,
            model,
            rho=args.rho,
            eta=args.eta,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "ASAM" in args.opt_type:
        minimizer = asam.ASAM(optimizer, model, rho=args.rho, eta=args.eta)
    elif "SAM" in args.opt_type:
        SAM = sam.SAM
        minimizer = SAM(optimizer, model, rho=args.rho, eta=args.eta,)
    else:
        raise NotImplemented

    return minimizer
