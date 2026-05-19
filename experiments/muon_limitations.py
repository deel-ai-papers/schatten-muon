"""
Muon Limitations Experiment

Compares Adam, Muon, and SMuon optimizers on deep linear networks
learning heavy-tailed singular value targets.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from smuon.optimizers.baseline import SingleDeviceMuonWithAuxAdam
from smuon.optimizers.adaptive import SingleDeviceSMuonWithAuxAdam
from smuon.wrap_model import ActivationRecorder


def random_orthogonal(n, device=None):
    """Return an n x n random orthogonal matrix (Haar measure)."""
    A = torch.randn(n, n, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


def matrix_heavy_tailed_singular_values(m, n, alpha, device=None):
    """
    Construct X = U S V^T with heavy-tailed singular values.
    - U (m x m), V (n x n): random orthogonal
    - S: diagonal with s_i = i^(-alpha) for i = 1, ..., min(m,n)
    """
    r = min(m, n)
    U = random_orthogonal(m, device)
    V = random_orthogonal(n, device)
    i = torch.arange(1, r + 1, device=device)
    s = i ** (-alpha)
    S = torch.zeros(m, n, device=device)
    S.diagonal().copy_(s)
    X = U @ S @ V.T
    return X


class MLP(nn.Module):
    def __init__(self, widths, use_activation=False):
        super().__init__()
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1], bias=False))
        self.layers = nn.ModuleList(layers)
        self.depth = len(widths) - 1
        self.use_activation = use_activation

    def forward(self, x):
        for i in range(self.depth):
            x = self.layers[i](x)
            if self.use_activation and i < self.depth - 1:
                x = torch.nn.functional.silu(x)
        return x


def create_optimizer(opt_name, model, lr):
    """Create optimizer by name."""
    if opt_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)

    elif opt_name == "Muon":
        # All params use Muon (matrices only in this MLP)
        muon_group = dict(
            params=list(model.parameters()),
            lr=lr,
            momentum=0.95,
            use_muon=True,
        )
        return SingleDeviceMuonWithAuxAdam([muon_group])

    elif opt_name == "SMuon":
        # All params use SMuon
        params = list(model.parameters())
        muon_group = dict(
            params=params,
            lr=lr,
            momentum=0.95,
            sv_momentum=0.9,
            beta2=0.999,
            use_muon=True,
        )
        adam_group = dict(
            params=[],
            lr=lr * 0.15,
            use_muon=False,
        )
        # Create param_names mapping for layerwise logging
        param_names = {p: f"layer_{i}" for i, p in enumerate(params)}
        return SingleDeviceSMuonWithAuxAdam(
            [muon_group, adam_group],
            pmin=1.02,
            pmax=10.0,
            init_p="pmax",
        )

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def run_experiment(
    opt_name,
    lr,
    depth,
    in_dim=100,
    out_dim=100,
    width=100,
    n_samples=1000,
    n_steps=1001,
    alpha=1.0,
    seed=42,
    verbose=True,
    smuon_interval=100,
):
    """Run a single training experiment.

    Parameters
    ----------
    opt_name : str
        Optimizer name: "Adam", "Muon", "SMuon", "MuonWithAdam", "SMuonWithAdam"
    lr : float
        Learning rate
    depth : int
        Number of layers in the MLP
    in_dim : int
        Input dimension
    out_dim : int
        Output dimension
    width : int
        Hidden layer width
    n_samples : int
        Number of training samples
    n_steps : int
        Number of training steps
    alpha : float
        Singular value decay exponent (higher = heavier tailed)
    seed : int
        Random seed
    verbose : bool
        Whether to print progress
    smuon_interval : int
        Interval for updating p-state in SMuon optimizers
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create teacher (heavy-tailed target)
    teacher_model = nn.Linear(in_dim, out_dim, bias=False).to(device)
    teacher_model.weight.data = matrix_heavy_tailed_singular_values(
        out_dim, in_dim, alpha=alpha, device=device
    )

    # Create student (deep linear network)
    student_model = MLP([in_dim] + [width] * (depth - 1) + [out_dim]).to(device)

    # Generate data
    inputs = torch.randn(n_samples, in_dim, device=device)
    labels = teacher_model(inputs).detach()

    # Create optimizer
    optimizer = create_optimizer(opt_name, student_model, lr)

    # Setup activation recording for SMuon variants
    use_smuon = opt_name in ("SMuon", "SMuonWithAdam")
    recorder = ActivationRecorder(student_model) if use_smuon else None

    losses = []
    p_stars = []  # Track p values over training

    for step in range(n_steps):
        optimizer.zero_grad()

        # Check if we should record activations for p-state update
        should_record = (
            use_smuon and (step % smuon_interval == 0)  # and (step != 0)
        )

        if should_record:
            with recorder.recording():
                outputs = student_model(inputs)
        else:
            outputs = student_model(inputs)

        loss = nn.functional.mse_loss(outputs, labels)
        loss.backward()

        # Update p-state with recorded activations
        if should_record:
            optimizer.update_p_state(activations=recorder.get_activations())
            recorder.clear()

            # Log p values
            if verbose:
                p_state = optimizer.get_p_state_for_logging()
                if p_state:
                    avg_p = np.mean([s["p_star"] for s in p_state.values()])
                    p_stars.append((step, avg_p))
                    # Print layerwise p_star values
                    print(f"  [Step {step}] Layerwise p_star values:")
                    for param_name, state in p_state.items():
                        print(f"    {param_name}: p_star = {state['p_star']:.4f}")

        optimizer.step()

        if verbose and step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6e}")

        losses.append(loss.item())

    # Cleanup
    if recorder is not None:
        recorder.remove_hooks()

    return losses


def run_sweep(
    opt_lrs_dict,
    depths,
    results_dir="./results_muon_sweep",
    n_steps=1001,
    seed=42,
    smuon_interval=100,
):
    """Run a sweep over optimizers, learning rates, and depths.

    Parameters
    ----------
    opt_lrs_dict : dict
        Dictionary mapping optimizer names to their learning rate lists.
        e.g. {"Adam": [1e-3, 1e-4], "Muon": [1e-2, 3e-3]}
    """
    os.makedirs(results_dir, exist_ok=True)

    # Store all results in memory for plotting
    all_results = {}

    for opt_name, lrs in opt_lrs_dict.items():
        all_results[opt_name] = {}
        for lr in lrs:
            for depth in depths:
                print(f"\n{'=' * 60}")
                print(f"Optimizer: {opt_name}, LR: {lr}, Depth: {depth}")
                print("=" * 60)

                losses = run_experiment(
                    opt_name=opt_name,
                    lr=lr,
                    depth=depth,
                    n_steps=n_steps,
                    seed=seed,
                    verbose=True,
                    smuon_interval=smuon_interval,
                )

                all_results[opt_name][(lr, depth)] = losses

    return all_results


def plot_results(
    all_results,
    depth,
    y_min=1e-6,
    y_max=1.0,
):
    """Plot loss curves for all optimizers at a given depth."""
    opt_names = list(all_results.keys())
    n_opts = len(opt_names)
    fig, axes = plt.subplots(1, n_opts, figsize=(5 * n_opts, 5))
    if n_opts == 1:
        axes = [axes]

    for ax, opt_name in zip(axes, opt_names):
        for (lr, d), losses in all_results[opt_name].items():
            if d == depth:
                ax.plot(losses, label=f"lr={lr}")
        ax.legend(loc="lower left")
        ax.set_yscale("log")
        ax.set_title(opt_name, fontsize=16)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

    plt.suptitle(f"Depth = {depth}", fontsize=20, y=1.02)
    plt.tight_layout()
    return fig


def plot_best_vs_depth(
    all_results,
    depths,
):
    """Plot best loss (over LRs) vs depth for each optimizer."""
    plt.figure(figsize=(8, 6))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for i, opt_name in enumerate(all_results.keys()):
        best_losses = []
        for depth in depths:
            best = float("inf")
            for (lr, d), losses in all_results[opt_name].items():
                if d == depth:
                    best = min(best, min(losses))
            best_losses.append(best if best != float("inf") else np.nan)
            print(f"{opt_name} Depth {depth}: best loss = {best:.2e}")

        plt.plot(
            depths,
            best_losses,
            f"{markers[i % len(markers)]}-",
            label=opt_name,
            linewidth=2,
            markersize=8,
        )

    plt.xlabel("Depth", fontsize=12)
    plt.ylabel("Best Loss (min over LRs and steps)", fontsize=12)
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    # Configuration - each optimizer has its own LR range
    opt_lrs_dict = {
        "SMuon": [1e-3, 1e-2, 2e-2],
        "Muon": [1e-3, 1e-2, 2e-2],
        "Adam": [3e-4, 1e-3, 2e-3],
    }
    depths = [1, 2, 5, 10]
    results_dir = "./figs/limitations"
    n_steps = 1001

    # Run the sweep
    print("Running optimizer sweep...")
    all_results = run_sweep(
        opt_lrs_dict=opt_lrs_dict,
        depths=depths,
        results_dir=results_dir,
        n_steps=n_steps,
    )

    # Plot results for each depth
    print("\nGenerating plots...")
    for depth in depths:
        fig = plot_results(all_results, depth)
        fig.savefig(
            f"{results_dir}/comparison_depth_{depth}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # Plot best loss vs depth
    fig = plot_best_vs_depth(all_results, depths)
    fig.savefig(f"{results_dir}/best_loss_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {results_dir}/")
