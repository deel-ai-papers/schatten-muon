#!/usr/bin/env python3
"""
Experiment script to compare optimal p* values computed by exact_tightness with different EMA and interval settings.
Compares three configurations:
1. EMA=0.0, interval=10
2. EMA=0.5, interval=40
3. EMA=0.9, interval=40
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
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 17
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReLUBatchNormMLP(nn.Module):
    def __init__(self, in_features=784, hidden_features=1024, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc_mid = nn.Linear(hidden_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden1 = F.relu(self.bn1(self.fc1(x)))
        hidden2 = F.relu(self.fc_mid(hidden1))
        return self.fc2(hidden2)


TRACKED = ["fc1.weight", "fc_mid.weight", "fc2.weight"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument(
        "--sv_momentum",
        type=float,
        default=0.2,
        help="EMA momentum for approx SV tracking. 0.0 = no smoothing.",
    )
    parser.add_argument("--seed", type=int, default=128)
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
            "weight_decay": 0.0,
        },
        {"params": adam_params, "use_muon": False, "lr": 3e-4, "betas": (0.95, 0.95)},
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

    # Three exact trackers with different EMA and smuon_interval configurations
    exact_ema_00 = create_p_approximator(
        method="exact_tightness",
        pmin=1.02,
        pmax=50.0,
        sv_momentum=0.0,
    )
    exact_ema_50 = create_p_approximator(
        method="exact_tightness",
        pmin=1.02,
        pmax=50.0,
        sv_momentum=0.5,
    )
    exact_ema_90 = create_p_approximator(
        method="exact_tightness",
        pmin=1.02,
        pmax=50.0,
        sv_momentum=0.9,
    )

    state_exact_00 = {n: {} for n in TRACKED}
    state_exact_50 = {n: {} for n in TRACKED}
    state_exact_90 = {n: {} for n in TRACKED}
    for name in TRACKED:
        exact_ema_00.create_state(state_exact_00[name])
        exact_ema_50.create_state(state_exact_50[name])
        exact_ema_90.create_state(state_exact_90[name])

    pstar_exact_00 = {n: [] for n in TRACKED}
    pstar_exact_50 = {n: [] for n in TRACKED}
    pstar_exact_90 = {n: [] for n in TRACKED}

    global_step = 0
    max_steps = 200
    interval_00 = 10
    interval_50 = 40
    interval_90 = 40

    model.train()
    print(f"Training for {max_steps} steps.")
    print(f"EMA=0.0 interval: {interval_00}, EMA=0.5 interval: {interval_50}, EMA=0.9 interval: {interval_90}")

    while global_step <= max_steps:
        for x, y in train_loader:
            if global_step > max_steps:
                break

            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()

            # Record when any of the three intervals is hit
            should_record = global_step > 0 and (
                global_step % interval_00 == 0 or
                global_step % interval_50 == 0 or
                global_step % interval_90 == 0
            )

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

                        # Update each tracker only at its interval
                        if global_step % interval_00 == 0:
                            p00, _ = exact_ema_00.update_and_compute_p(
                                state_exact_00[name],
                                grad_2d,
                                act_clone,
                                use_gram=False,
                                nesterov=False,
                                **kw,
                            )
                            pstar_exact_00[name].append(p00)

                        if global_step % interval_50 == 0:
                            p50, _ = exact_ema_50.update_and_compute_p(
                                state_exact_50[name],
                                grad_2d,
                                act_clone,
                                use_gram=False,
                                nesterov=False,
                                **kw,
                            )
                            pstar_exact_50[name].append(p50)

                        if global_step % interval_90 == 0:
                            p90, _ = exact_ema_90.update_and_compute_p(
                                state_exact_90[name],
                                grad_2d,
                                act_clone,
                                use_gram=False,
                                nesterov=False,
                                **kw,
                            )
                            pstar_exact_90[name].append(p90)

                recorder.clear()

                # Track steps for each configuration
                if global_step % interval_00 == 0:
                    if "steps_00" not in locals():
                        steps_00 = []
                    steps_00.append(global_step)
                if global_step % interval_50 == 0:
                    if "steps_50" not in locals():
                        steps_50 = []
                    steps_50.append(global_step)
                if global_step % interval_90 == 0:
                    if "steps_90" not in locals():
                        steps_90 = []
                    steps_90.append(global_step)

            optimizer.step()
            global_step += 1

    # Ensure all step lists exist
    if "steps_00" not in locals():
        steps_00 = []
    if "steps_50" not in locals():
        steps_50 = []
    if "steps_90" not in locals():
        steps_90 = []

    # ── Plotting ─────────────────────────────────────────────────────────────
    def ylim(ex00, ex50, ex90):
        all_v = ex00 + ex50 + ex90
        if not all_v:
            return (1, 50)
        return (max(min(all_v) - 1, 1), min(max(all_v) + 1, 50))

    style_00 = dict(
        marker="o", color="black", lw=2, ms=7, label=r"Exact $p^*$, EMA=0.0, interval=10"
    )
    style_50 = dict(
        marker="s",
        color="darkorange",
        lw=2,
        ms=7,
        ls="-.",
        label=r"Exact $p^*$, EMA=0.5, interval=40",
    )
    style_90 = dict(
        marker="D",
        color="forestgreen",
        lw=2,
        ms=7,
        ls=":",
        label=r"Exact $p^*$, EMA=0.9, interval=40",
    )

    layer_titles = {
        "fc1.weight": "Input Layer",
        "fc_mid.weight": "Middle Layer",
        "fc2.weight": "Output Layer",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, name in zip(axes, TRACKED):
        if steps_00:
            ax.plot(steps_00, pstar_exact_00[name], **style_00)
        if steps_50:
            ax.plot(steps_50, pstar_exact_50[name], **style_50)
        if steps_90:
            ax.plot(steps_90, pstar_exact_90[name], **style_90)

        ax.set_xlabel("Training Step", fontsize=15)
        ax.set_title(layer_titles[name], fontsize=17)
        ax.set_ylim(
            *ylim(
                pstar_exact_00[name],
                pstar_exact_50[name],
                pstar_exact_90[name],
            )
        )
        ax.grid(True, alpha=0.3)

        # Fixed overlapping xticks by setting them every 50
        ax.set_xticks(np.arange(0, max_steps + 1, 50))

    axes[0].set_ylabel(r"$p^*$", fontsize=15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_path = "figs/exact_p_ema_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
