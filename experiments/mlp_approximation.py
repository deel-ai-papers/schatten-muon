#!/usr/bin/env python3
"""
Experiment script to compare optimal p* values computed by exact_tightness vs approx_tightness.
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from smuon.wrap_model import ActivationRecorder
from smuon.optimizers.adaptive import SingleDeviceSMuonWithAuxAdam
from smuon.svs.p_registry import create_p_approximator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "serif"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReLUBatchNormMLP(nn.Module):
    def __init__(self, in_features=784, hidden_features=500, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc_mid = nn.Linear(hidden_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden1 = F.relu(self.bn1(self.fc1(x)))
        hidden2 = F.relu(self.fc_mid(hidden1))
        return self.fc2(hidden2)


TRACKED = ["fc1.weight", "fc_mid.weight", "fc2.weight"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument(
        "--sv_momentum",
        type=float,
        default=0.2,
        help="EMA momentum for approx SV tracking. 0.0 = no smoothing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1,), (0.1,))]
    )
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    model = ReLUBatchNormMLP().to(args.device)
    criterion = nn.CrossEntropyLoss()

    muon_params, adam_params, param_names = [], [], {}
    for name, param in model.named_parameters():
        param_names[param] = name
        (muon_params if param.ndim >= 2 else adam_params).append(param)

    param_groups = [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": 0.001,
            "momentum": args.beta1,
            "sv_momentum": 0.0,
            "weight_decay": 0.05,
        },
        {"params": adam_params, "use_muon": False, "lr": 3e-4, "betas": (0.9, 0.95)},
    ]
    optimizer = SingleDeviceSMuonWithAuxAdam(
        param_groups,
        param_names=param_names,
        pmin=1.02,
        pmax=50.0,
        p_method="exact_tightness",
        subsampling_ratio=1.0,
    )

    recorder = ActivationRecorder(model, use_gram=False)

    # Three approximate trackers at different subsampling ratios
    approx_app_30 = create_p_approximator(
        method="approx_tightness",
        pmin=1.02,
        pmax=50.0,
        subsampling_ratio=0.3,
        sv_momentum=0.85,
    )
    approx_app_60 = create_p_approximator(
        method="approx_tightness",
        pmin=1.02,
        pmax=50.0,
        subsampling_ratio=0.60,
        sv_momentum=0.75,
    )
    approx_app_80 = create_p_approximator(
        method="approx_tightness",
        pmin=1.02,
        pmax=50.0,
        subsampling_ratio=0.80,
        sv_momentum=0.65,  # Lower momentum for higher ratio
    )

    state_approx_30 = {n: {} for n in TRACKED}
    state_approx_60 = {n: {} for n in TRACKED}
    state_approx_80 = {n: {} for n in TRACKED}
    for name in TRACKED:
        approx_app_30.create_state(state_approx_30[name])
        approx_app_60.create_state(state_approx_60[name])
        approx_app_80.create_state(state_approx_80[name])

    steps_recorded = []
    pstar_exact = {n: [] for n in TRACKED}
    pstar_ap_30 = {n: [] for n in TRACKED}
    pstar_ap_60 = {n: [] for n in TRACKED}
    pstar_ap_80 = {n: [] for n in TRACKED}

    global_step = 0
    max_steps = 200
    record_interval = 10

    model.train()
    print(f"Training for {max_steps} steps, tracking p* every {record_interval} steps.")

    while global_step <= max_steps:
        for x, y in train_loader:
            if global_step > max_steps:
                break

            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()

            should_record = global_step % record_interval == 0 and global_step > 0

            if should_record:
                with recorder.recording():
                    outputs = model(x)
            else:
                outputs = model(x)

            criterion(outputs, y).backward()

            if should_record:
                acts = recorder.get_activations()

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name not in TRACKED:
                            continue
                        if param.grad is None or param not in acts:
                            continue

                        act_clone = acts[param].clone()
                        grad_2d = param.grad.view(param.grad.size(0), -1)

                        opt_state = optimizer.state[param]
                        mom_buf = opt_state.get(
                            "momentum_buffer", torch.zeros_like(param.grad)
                        )
                        muon_step = opt_state.get("muon_step", 0) + 1
                        bc = 1.0 - args.beta1**muon_step
                        mom_2d = (mom_buf / bc).reshape(mom_buf.size(0), -1)

                        kw = dict(
                            mom_2d=mom_2d,
                            beta1=args.beta1,
                            layer_name=name,
                            step=global_step,
                        )

                        p30, _ = approx_app_30.update_and_compute_p(
                            state_approx_30[name],
                            grad_2d,
                            act_clone,
                            use_gram=False,
                            nesterov=False,
                            **kw,
                        )
                        pstar_ap_30[name].append(p30)

                        p60, _ = approx_app_60.update_and_compute_p(
                            state_approx_60[name],
                            grad_2d,
                            act_clone,
                            use_gram=False,
                            nesterov=False,
                            **kw,
                        )
                        pstar_ap_60[name].append(p60)

                        p80, _ = approx_app_80.update_and_compute_p(
                            state_approx_80[name],
                            grad_2d,
                            act_clone,
                            use_gram=False,
                            nesterov=False,
                            **kw,
                        )
                        pstar_ap_80[name].append(p80)

                optimizer.update_p_state(activations=acts, use_gram=False)
                recorder.clear()

                log_dict = optimizer.get_p_state_for_logging()
                for name in TRACKED:
                    if name in log_dict:
                        pstar_exact[name].append(log_dict[name]["p_star"])

                steps_recorded.append(global_step)

            optimizer.step()
            global_step += 1

    # ── Plotting ─────────────────────────────────────────────────────────────
    def ylim(exact, ap30, ap60, ap80):
        all_v = exact + ap30 + ap60 + ap80
        if not all_v:
            return (1, 50)
        return (max(min(all_v) - 1, 1), min(max(all_v) + 1, 50))

    style_exact = dict(
        marker="o", color="black", lw=2, ms=7, label=r"Exact $p^*$, no EMA"
    )
    style_80 = dict(
        marker="D",
        color="forestgreen",
        lw=2,
        ms=7,
        ls=":",
        label=r"Approx. $p^*$, $r=0.8$, EMA=0.2",
    )
    style_60 = dict(
        marker="s",
        color="royalblue",
        lw=2,
        ms=7,
        ls="--",
        label=r"Approx. $p^*$, $r=0.6$, EMA=0.2",
    )
    style_30 = dict(
        marker="^",
        color="darkorange",
        lw=2,
        ms=7,
        ls="-.",
        label=r"Approx. $p^*$, $r=0.3$, EMA=0.2",
    )

    layer_titles = {
        "fc1.weight": "Input Layer",
        "fc_mid.weight": "Middle Layer",
        "fc2.weight": "Output Layer",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, name in zip(axes, TRACKED):
        if steps_recorded:
            ax.plot(steps_recorded, pstar_exact[name], **style_exact)
            ax.plot(steps_recorded, pstar_ap_80[name], **style_80)
            ax.plot(steps_recorded, pstar_ap_60[name], **style_60)
            ax.plot(steps_recorded, pstar_ap_30[name], **style_30)

        ax.set_xlabel("Training Step", fontsize=11.88)
        ax.set_title(layer_titles[name], fontsize=14)
        ax.set_ylim(
            *ylim(
                pstar_exact[name],
                pstar_ap_30[name],
                pstar_ap_60[name],
                pstar_ap_80[name],
            )
        )
        ax.grid(True, alpha=0.3)

        # Fixed overlapping xticks by setting them every 50
        ax.set_xticks(np.arange(0, max_steps + 1, 50))

    axes[0].set_ylabel(r"$p^*$", fontsize=13.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=13,
    )
    # plt.suptitle(
    #     f"Power-law tail model  (w/ EMA={args.sv_momentum})",
    #     fontsize=13, y=1.01
    # )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_path = "figs/p_approximation_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
