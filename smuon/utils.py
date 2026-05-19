import torch
from smuon.optimizers.baseline import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from smuon.optimizers.adaptive import SMuonWithAuxAdam, SingleDeviceSMuonWithAuxAdam
from smuon.optimizers.exact import (
    ExactSMuonWithAuxAdam,
    ExactMuonWithOptimalPLogging,
)
from smuon.optimizers.newton import NewtonMuon


def get_muon_groups(model, args):
    def cond(p):
        if p.ndim in [1, 3]:
            return False

        size_ok = sum([p.size(i) > 1 for i in range(p.ndim)]) >= 2
        return size_ok

    muon_params = [p for n, p in model.named_parameters() if cond(p)]
    adam_params = [p for n, p in model.named_parameters() if not cond(p)]
    muon_wd = getattr(args, "muon_wd", 0.0)
    adam_wd = getattr(args, "adam_wd", 0.0)
    list_params = [
        dict(
            params=muon_params,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=muon_wd,
            use_muon=True,
        ),
        dict(
            params=adam_params,
            lr=args.adam_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=adam_wd,
            use_muon=False,
        ),
    ]
    return list_params


def get_smuon_groups(model, args):
    def cond(p):
        if p.ndim in [1, 3]:
            return False

        size_ok = sum([p.size(i) > 1 for i in range(p.ndim)]) >= 2
        return size_ok

    muon_params = [p for n, p in model.named_parameters() if cond(p)]
    adam_params = [p for n, p in model.named_parameters() if not cond(p)]
    param_names = {p: n for n, p in model.named_parameters()}
    muon_wd = getattr(args, "muon_wd", 0.0)
    adam_wd = getattr(args, "adam_wd", 0.0)
    list_params = [
        dict(
            params=muon_params,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            sv_momentum=args.sv_momentum,
            beta2=args.beta2,
            eps=getattr(args, "eps", 1e-8),
            weight_decay=muon_wd,
            use_muon=True,
        ),
        dict(
            params=adam_params,
            lr=args.adam_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=adam_wd,
            use_muon=False,
        ),
    ]
    return list_params, param_names


def _build_smuon_kwargs(model, args):
    """Shared kwarg construction for SMuon-family optimizers."""
    param_groups, param_names = get_smuon_groups(model, args)

    return dict(
        param_groups=param_groups,
        param_names=param_names,
        pmin=getattr(args, "pmin"),
        pmax=getattr(args, "pmax"),
        eps=getattr(args, "eps", 1e-5),
        init_p=getattr(args, "init_p", "pmax"),
        p_method=getattr(args, "p_method", "exact_momentum"),
        subsampling_ratio=getattr(args, "subsampling_ratio", 0.1),
        moment_type=getattr(args, "moment_type", "none"),
    )


def get_optimizer(model, args):
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.adam_lr)
    elif args.optimizer == "sgd":
        sgd_lr = getattr(args, "sgd_lr", 1e-2)
        sgd_momentum = getattr(args, "sgd_momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=sgd_lr, momentum=sgd_momentum
        )
    elif args.optimizer == "muon":
        optimizer = MuonWithAuxAdam(get_muon_groups(model, args))
    elif args.optimizer == "smuon":
        kw = _build_smuon_kwargs(model, args)
        optimizer = SMuonWithAuxAdam(
            **{k: v for k, v in kw.items() if k != "param_groups"},
            param_groups=kw["param_groups"],
        )
    elif args.optimizer == "exact-smuon":
        # Diagnostic: exact U Σ^{1/p*} V^T via full SVD (no Newton-Schulz /
        # Taylor approximation). Slow; use for ablations only. Single-device
        # even under distributed launch.
        kw = _build_smuon_kwargs(model, args)
        optimizer = ExactSMuonWithAuxAdam(
            **{k: v for k, v in kw.items() if k != "param_groups"},
            param_groups=kw["param_groups"],
        )
    elif args.optimizer == "muon-plogging":
        # Diagnostic: apply exact Muon updates (polar factor U V^T), but
        # still compute and log p* at every update_p_state call. Tests
        # whether Muon's update trajectory drives measured p* upward.
        kw = _build_smuon_kwargs(model, args)
        optimizer = ExactMuonWithOptimalPLogging(
            **{k: v for k, v in kw.items() if k != "param_groups"},
            param_groups=kw["param_groups"],
        )
    elif args.optimizer == "newton-muon":
        kw = _build_smuon_kwargs(model, args)
        optimizer = NewtonMuon(
            **{k: v for k, v in kw.items() if k != "param_groups"},
            param_groups=kw["param_groups"],
        )
    else:
        raise ValueError(f"Unrecognized optimizer name: {args.optimizer}")
    return optimizer


def get_optimizer_single_gpu(model, args):
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.adam_lr)
    elif args.optimizer == "sgd":
        sgd_lr = getattr(args, "sgd_lr", 1e-2)
        sgd_momentum = getattr(args, "sgd_momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=sgd_lr, momentum=sgd_momentum
        )
    elif args.optimizer == "muon":
        optimizer = SingleDeviceMuonWithAuxAdam(get_muon_groups(model, args))
    elif args.optimizer == "smuon":
        kw = _build_smuon_kwargs(model, args)
        optimizer = SingleDeviceSMuonWithAuxAdam(
            **{k: v for k, v in kw.items() if k != "param_groups"},
            param_groups=kw["param_groups"],
        )
    elif args.optimizer == "exact-smuon":
        kw = _build_smuon_kwargs(model, args)
        optimizer = ExactSMuonWithAuxAdam(
            **{k: v for k, v in kw.items() if k != "param_groups"},
            param_groups=kw["param_groups"],
        )
    elif args.optimizer == "muon-plogging":
        kw = _build_smuon_kwargs(model, args)
        optimizer = ExactMuonWithOptimalPLogging(
            **{k: v for k, v in kw.items() if k != "param_groups"},
            param_groups=kw["param_groups"],
        )
    elif args.optimizer == "newton-muon":
        kw = _build_smuon_kwargs(model, args)
        optimizer = NewtonMuon(
            **{k: v for k, v in kw.items() if k != "param_groups"},
            param_groups=kw["param_groups"],
        )
    else:
        raise ValueError(f"Unrecognized optimizer name: {args.optimizer}")
    return optimizer
