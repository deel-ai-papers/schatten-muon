"""
Benchmark matrix p-th root approximation for varying p values.

Compares:
1. linf_remez (smuon.coeffs.linf_remez) -- dynamic SLSQP layer-wise coefficients
2. Taylor approximation (smuon.p_root.fractional_power_from_polar)

Tests p values: 2.0, 4.0, 10, 15, 30 to evaluate performance at increasing p.
Compares approximation quality (Relative Frobenius error) vs SVD baseline.
"""

import argparse
import time
from typing import Any, Callable, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from smuon.coeffs.linf_remez import solve_fixed
from smuon.p_root import fractional_power_from_polar
from smuon.cholesky import batched_fractional_polar_zolotarev


# ============================================================================
# Matplotlib configuration
# ============================================================================

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "DejaVu Serif",
    "Times New Roman",
    "Liberation Serif",
]
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams["xtick.labelsize"] = 9
matplotlib.rcParams["ytick.labelsize"] = 9
matplotlib.rcParams["legend.fontsize"] = 9


# Shared epsilon used across methods for the spectrum lower bound.
GLOBAL_EPSILON = 1e-2


# ============================================================================
# Zolotarev coefficient generation
# ============================================================================


def generate_quadrature_coeffs(p, k, epsilon=GLOBAL_EPSILON):
    """Zolotarev-type coefficients for x^(-alpha), alpha = (1-p)/2, via
    Gauss-Legendre quadrature on the integral representation."""
    if p == 1.0:
        return 1.0, np.array([]), np.array([])

    alpha = (1.0 - p) / 2.0
    u, v = np.polynomial.legendre.leggauss(k)
    L = np.sqrt(epsilon)
    shifts = L * (1 + u) / (1 - u)
    dt_du = 2 * L / ((1 - u) ** 2)
    sin_term = np.sin(alpha * np.pi) / np.pi
    weights = sin_term * v * dt_du * (shifts**-alpha)
    C0 = 0.0
    return C0, weights, shifts


# ============================================================================
# Method wrappers
# ============================================================================


def linf_remez_pth_root(X, coeffs, p=2):
    """Compute U Sigma^{1/p} V^T using linf_remez Newton-Schulz coefficients."""
    if coeffs is None:
        return X
    orig_dtype = X.dtype
    X_f = X.float()
    transpose = X_f.size(-2) > X_f.size(-1)
    if transpose:
        X_f = X_f.mT
    scale = X_f.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7
    X_f = X_f / scale
    for a, b, c in coeffs:
        A = X_f @ X_f.mT
        X_f = a * X_f + (b * A + c * A @ A) @ X_f
    if transpose:
        X_f = X_f.mT
    return (X_f * (scale ** (1.0 / p))).to(orig_dtype)


def taylor_pth_root(X, p=2, max_extra_iters=5, epsilon=GLOBAL_EPSILON):
    """Compute U Sigma^{1/p} V^T via polar factor + adaptive Taylor."""
    mat, _ = fractional_power_from_polar(
        X, p=1.0 / p, eigen_floor=epsilon, max_extra_iters=max_extra_iters
    )
    return mat


def zolotarev_pth_root(X, p=2, k=10, epsilon=GLOBAL_EPSILON):
    """Compute U Sigma^{1/p} V^T via Zolotarev approximation + batched Cholesky."""
    p_order = 1.0 / p
    C0_np, weights_np, shifts_np = generate_quadrature_coeffs(p_order, k, epsilon)
    C0 = float(C0_np)
    weights = torch.tensor(weights_np, device=X.device, dtype=X.dtype)
    shifts = torch.tensor(shifts_np, device=X.device, dtype=X.dtype)
    coeffs = (C0, weights, shifts)
    return batched_fractional_polar_zolotarev(X, p_order, coeffs)


# ============================================================================
# Exact baseline
# ============================================================================


def exact_svd_pth_root(X, p=2):
    """Exact U Sigma^{1/p} V^T via SVD."""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S_p = S.pow(1.0 / p)
    return U @ torch.diag(S_p) @ Vh


# ============================================================================
# Method registry
# ============================================================================


METHOD_CONFIG: Dict[str, Dict[str, Any]] = {
    "linf_remez": {
        "label": "Newton-Schulz w/ Remez Coefficients",
        "color": "C0",
        "marker": "o",
    },
    "taylor": {
        "label": "Taylor Approximation from Polar Factor",
        "color": "C3",
        "marker": "D",
    },
    "zolotarev": {
        "label": "Cholesky Factorization and Zolotarev Approximation",
        "color": "C2",
        "marker": "^",
    },
}


def _run_method(
    method: str, X: torch.Tensor, p: float, budget: int, coeffs_cache: Dict[int, Any]
) -> torch.Tensor:
    if method == "linf_remez":
        return linf_remez_pth_root(X, coeffs_cache[budget], p=p)
    if method == "taylor":
        return taylor_pth_root(X, p=p, max_extra_iters=budget, epsilon=GLOBAL_EPSILON)
    if method == "zolotarev":
        return zolotarev_pth_root(X, p=p, k=budget, epsilon=GLOBAL_EPSILON)
    raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Matrix generation
# ============================================================================


def generate_test_matrix(matrix_size, device, vary_condition=False):
    """Generate a test matrix, optionally with varying condition number."""
    if not vary_condition:
        return torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)

    U, _ = torch.linalg.qr(
        torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    )
    V, _ = torch.linalg.qr(
        torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    )
    # Log-uniform condition number in [10, 1000]
    kappa = 10 ** (1 + torch.rand(1, device=device).item() * 2)
    S = torch.logspace(
        1.0, 1.0 / kappa, matrix_size, device=device, dtype=torch.float32
    )
    return U @ torch.diag(S) @ V.T


# ============================================================================
# Plotting
# ============================================================================


def plot_coefficient_approximation(ax, p, eps=GLOBAL_EPSILON):
    """Show the scalar polynomial quality for each compositional method."""
    p_order = 1.0 / p

    # linf_remez, 5 iterations
    coeffs_remez = solve_fixed(num_steps=5, p_order=p_order, eps=eps)

    # Taylor order 5 around x = 1
    max_k = 5
    taylor_c = [1.0]
    for j in range(1, max_k + 1):
        taylor_c.append(taylor_c[-1] * (p_order - j + 1) / j)

    x = np.linspace(eps, 1.0, 1000)
    y_true = x**p_order

    def compose(coeffs, x):
        y = x.copy()
        for a, b, c in coeffs:
            y = a * y + b * y**3 + c * y**5
        return y

    y_remez = compose(coeffs_remez, x) if coeffs_remez is not None else x

    y_taylor = np.zeros_like(x)
    diff = x - 1.0
    for j in range(max_k + 1):
        y_taylor += taylor_c[j] * (diff**j)

    ax.plot(x, y_true, "k-", linewidth=2, label=rf"$x^{{1/{p}}}$ (exact)")
    ax.plot(x, y_remez, "C0--", linewidth=2, label="Remez Polynomial")
    ax.plot(x, y_taylor, "C3:", linewidth=2, label="Taylor Approximation")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_xlim(left=eps)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Scalar polynomial quality (p = ${p:.0f}$ | 5 iters)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_error_vs_svd(
    ax,
    p,
    matrix_size=1024,
    n_trials=25,
    vary_condition=False,
):
    """Plot Relative Frobenius error comparing approximations to SVD baseline."""
    budgets = list(range(1, 7))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_order = 1.0 / p
    coeffs_cache = {
        b: solve_fixed(num_steps=b, p_order=p_order, eps=GLOBAL_EPSILON)
        for b in budgets
    }

    # Store results for each method
    results = {method: {"errors": [], "errors_std": []} for method in METHOD_CONFIG}

    for budget in tqdm(budgets, desc=f"p={p}", leave=False):
        for method in METHOD_CONFIG:
            trial_errors = []

            for _ in range(n_trials):
                X = generate_test_matrix(
                    matrix_size, device, vary_condition=vary_condition
                )
                with torch.no_grad():
                    X_true = exact_svd_pth_root(X, p=p)
                    try:
                        X_approx = _run_method(method, X, p, budget, coeffs_cache)
                        err = torch.norm(X_approx - X_true, p="fro") / torch.norm(
                            X_true, p="fro"
                        )
                        trial_errors.append(err.item())
                    except Exception:
                        trial_errors.append(np.nan)

            results[method]["errors"].append(np.nanmean(trial_errors))
            if vary_condition:
                results[method]["errors_std"].append(np.nanstd(trial_errors))

    # Plot results
    for method, cfg in METHOD_CONFIG.items():
        if vary_condition:
            ax.errorbar(
                budgets,
                results[method]["errors"],
                yerr=results[method]["errors_std"],
                marker=cfg["marker"],
                label=cfg["label"],
                color=cfg["color"],
                capsize=4,
                elinewidth=1.5,
                capthick=1.5,
                linewidth=2,
                markersize=7,
            )
        else:
            ax.plot(
                budgets,
                results[method]["errors"],
                marker=cfg["marker"],
                label=cfg["label"],
                color=cfg["color"],
                linewidth=2,
                markersize=7,
            )

    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title(f"Approximation error vs SVD (p = ${p:.0f}$)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark matrix p-th root for varying p values"
    )
    parser.add_argument(
        "--other_distrib",
        action="store_true",
        help="Use matrices with varying condition numbers (log-uniform in [10, 1000]) "
        "instead of standard Gaussian; enables error bars on plots.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=25,
        help="Number of random trials per (method, budget) cell.",
    )
    parser.add_argument(
        "--matrix_size",
        type=int,
        default=1024,
        help="Matrix size for benchmarks.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figs/maximum_p.png",
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    if args.other_distrib:
        print("Distribution: log-uniform condition number in [10, 1000]")
    else:
        print("Distribution: standard Gaussian")

    # Test p values
    p_values = [2.0, 4.0, 10.0, 15.0, 30.0, 50.0]
    n_plots = len(p_values)

    # Create figure with 2 columns (left: coeffs, right: error) and one row per p value
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 5 * n_plots))

    for idx, p in enumerate(p_values):
        print(f"\n=== p = {p} (computing x^{{1/{p}}}) ===")

        # Left plot: coefficient approximation
        plot_coefficient_approximation(axes[idx, 0], p=p)

        # Right plot: error vs SVD
        plot_error_vs_svd(
            axes[idx, 1],
            p=p,
            matrix_size=args.matrix_size,
            n_trials=args.n_trials,
            vary_condition=args.other_distrib,
        )

    # Setup the single horizontal legend at the bottom
    handles, labels = axes[0, 1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=12,
        markerscale=1.5,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
