"""
Benchmark approximation quality and runtime for matrix p-th root methods.

Compares:
1. linf_remez (smuon.coeffs.linf_remez) -- dynamic SLSQP layer-wise coefficients
2. Qi et al., 2026 -- coupled Newton-Schulz
3. Zolotarev -- quadrature + batched Cholesky
4. Polar + Taylor (smuon.p_root.fractional_power_from_polar)

Plots benchmark exact runtime vs Relative Frobenius error, displaying a shared
Pareto frontier instead of theoretical matmul counts.
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
from smuon.cholesky import batched_fractional_polar_zolotarev
from smuon.p_root import fractional_power_from_polar


# ============================================================================
# Matplotlib configuration
# ============================================================================

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "DejaVu Serif",
    "Times New Roman",
    "Liberation Serif",
]
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["axes.labelsize"] = 17
matplotlib.rcParams["axes.titlesize"] = 17
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 12


# Shared epsilon used across methods for the spectrum lower bound.
GLOBAL_EPSILON = 1e-2


# ============================================================================
# Reference implementation from Qi et al, 2026
# ============================================================================


def newton_schulz_sqrt(A, K=10):
    """Coupled Newton-Schulz: returns A^{1/2}, A^{-1/2} for symmetric PD A."""
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    alpha = torch.norm(A, "fro")
    Y, Z = A / alpha, I.clone()
    for _ in range(K):
        T = 3 * I - Z @ Y
        Y, Z = 0.5 * Y @ T, 0.5 * T @ Z
    return alpha.sqrt() * Y, Z / alpha.sqrt()


def matrix_pth_root(X, p=2, K=10):
    """Compute U Sigma^{1/p} V^T for p in {2, 4} via coupled Newton-Schulz."""
    A = X.T @ X  # V Sigma^2 V^T
    S, S_inv = newton_schulz_sqrt(A, K)  # V Sigma V^T
    polar = X @ S_inv  # U V^T
    # repeatedly halve the exponent: Sigma -> Sigma^{1/2} -> Sigma^{1/4}
    for _ in range({2: 1, 4: 2}[p]):
        S, _ = newton_schulz_sqrt(S, K)
    return polar @ S


# ============================================================================
# Method wrappers
# ============================================================================


def our_matrix_pth_root(X, coeffs, p=2):
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


def p_root_approx(X, p=2, max_extra_iters=5, epsilon=GLOBAL_EPSILON):
    """Compute U Sigma^{1/p} V^T via polar factor + adaptive Taylor."""
    mat, _ = fractional_power_from_polar(
        X, p=1.0 / p, eigen_floor=epsilon, max_extra_iters=max_extra_iters
    )
    return mat


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
    "ours": {
        "label": "Newton-Schulz w/ Remez Coefficients",
        "color": "C0",
        "marker": "o",
    },
    "qi": {
        "label": "Coupled Newton-Schulz (Qi et al., 2026)",
        "color": "C1",
        "marker": "s",
    },
    "zolotarev": {
        "label": "Cholesky Factorization and Zolotarev Approximation",
        "color": "C2",
        "marker": "^",
    },
    "p_root": {
        "label": "Taylor Approximation from Polar Factor",
        "color": "C3",
        "marker": "D",
    },
}


def _run_method(
    method: str, X: torch.Tensor, p: int, budget: int, coeffs_cache: Dict[int, Any]
) -> torch.Tensor:
    if method == "ours":
        return our_matrix_pth_root(X, coeffs_cache[budget], p=p)
    if method == "qi":
        return matrix_pth_root(X, p=p, K=budget)
    if method == "zolotarev":
        return zolotarev_pth_root(X, p=p, k=budget, epsilon=GLOBAL_EPSILON)
    if method == "p_root":
        return p_root_approx(X, p=p, max_extra_iters=budget, epsilon=GLOBAL_EPSILON)
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
    coeffs_ours = solve_fixed(num_steps=5, p_order=p_order, eps=eps)

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

    y_ours = compose(coeffs_ours, x)

    y_taylor = np.zeros_like(x)
    diff = x - 1.0
    for j in range(max_k + 1):
        y_taylor += taylor_c[j] * (diff**j)

    ax.plot(x, y_true, "k-", linewidth=2, label=rf"$x^{{1/{p}}}$ (exact)")
    ax.plot(x, y_ours, "C0--", linewidth=2, label="Remez Polynomial")
    ax.plot(x, y_taylor, "C3:", linewidth=2, label="Taylor Approximation")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_xlim(left=eps)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Scalar polynomial quality (p = ${p}$ | 5 iters)")
    ax.legend(fontsize=15)  # Slightly enlarged this legend too
    ax.grid(True, alpha=0.3)


def plot_error_vs_runtime(
    ax,
    p,
    matrix_size_err=2048,
    matrix_size_time=1024,
    n_trials=25,
    vary_condition=False,
):
    """Plot Relative Frobenius error (y) vs Wall-clock runtime (x)."""
    budgets = list(range(1, 7))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_order = 1.0 / p
    coeffs_cache = {
        b: solve_fixed(num_steps=b, p_order=p_order, eps=GLOBAL_EPSILON)
        for b in budgets
    }

    # Helper function for highly precise PyTorch benchmarking
    def benchmark_fn(func, *args, **kwargs):
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            func(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event) / 1000.0  # convert ms to seconds
        else:
            t0 = time.perf_counter()
            func(*args, **kwargs)
            return time.perf_counter() - t0

    # SVD baseline benchmark
    svd_times = []
    # 1. Warmup for SVD
    for _ in range(5):
        X_warm = generate_test_matrix(
            matrix_size_time, device, vary_condition=vary_condition
        )
        with torch.no_grad():
            exact_svd_pth_root(X_warm, p=p)

    # 2. Benchmark SVD
    for _ in tqdm(range(n_trials), desc=f"SVD baseline p={p}", leave=False):
        X = generate_test_matrix(
            matrix_size_time, device, vary_condition=vary_condition
        )
        with torch.no_grad():
            elapsed = benchmark_fn(exact_svd_pth_root, X, p=p)
        svd_times.append(elapsed)

    svd_mean = np.mean(svd_times)
    ax.axvline(svd_mean, linestyle="--", color="gray", alpha=0.7, label="SVD (exact)")
    if vary_condition:
        svd_std = np.std(svd_times)
        ax.axvspan(svd_mean - svd_std, svd_mean + svd_std, color="gray", alpha=0.15)

    # Iterative methods benchmark
    for method, cfg in METHOD_CONFIG.items():
        xs_mean, ys_mean = [], []
        xs_std, ys_std = [], []

        for budget in budgets:
            trial_times = []
            trial_errors = []

            # WARMUP: Specific to this method and this budget
            for _ in range(3):
                X_warm = generate_test_matrix(
                    matrix_size_time, device, vary_condition=vary_condition
                )
                try:
                    with torch.no_grad():
                        _ = _run_method(method, X_warm, p, budget, coeffs_cache)
                except Exception:
                    pass

            # Timing benchmark using CUDA Events
            for _ in range(n_trials):
                X = generate_test_matrix(
                    matrix_size_time, device, vary_condition=vary_condition
                )
                try:
                    with torch.no_grad():
                        elapsed = benchmark_fn(
                            _run_method, method, X, p, budget, coeffs_cache
                        )
                    trial_times.append(elapsed)
                except Exception:
                    trial_times.append(np.nan)

            # Error benchmark
            for _ in range(n_trials):
                X = generate_test_matrix(
                    matrix_size_err, device, vary_condition=vary_condition
                )
                with torch.no_grad():
                    X_true = exact_svd_pth_root(X, p=p)
                try:
                    with torch.no_grad():
                        X_approx = _run_method(method, X, p, budget, coeffs_cache)
                    err = torch.norm(X_approx - X_true, p="fro") / torch.norm(
                        X_true, p="fro"
                    )
                    trial_errors.append(err.item())
                except Exception:
                    trial_errors.append(np.nan)

            xs_mean.append(np.nanmean(trial_times))
            ys_mean.append(np.nanmean(trial_errors))

            if vary_condition:
                xs_std.append(np.nanstd(trial_times))
                ys_std.append(np.nanstd(trial_errors))

        # Plotting results for the method
        if vary_condition:
            ax.errorbar(
                xs_mean,
                ys_mean,
                xerr=xs_std,
                yerr=ys_std,
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
                xs_mean,
                ys_mean,
                marker=cfg["marker"],
                label=cfg["label"],
                color=cfg["color"],
                linewidth=2,
                markersize=7,
            )

    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title(f"Error vs Runtime (p = ${p}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark matrix p-th root approximation methods"
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
        "--matrix_size_err",
        type=int,
        default=1024,
        help="Matrix size used for the error plots.",
    )
    parser.add_argument(
        "--matrix_size_time",
        type=int,
        default=1024,
        help="Matrix size used for the timing plots.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figs/matrix_approximation_benchmark.png",
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    if args.other_distrib:
        print("Distribution: log-uniform condition number in [10, 1000]")
    else:
        print("Distribution: standard Gaussian")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, p in enumerate([2, 4]):
        print(f"\n=== p = {p} (computing x^{{1/{p}}}) ===")
        plot_coefficient_approximation(axes[row, 0], p=p)
        plot_error_vs_runtime(
            axes[row, 1],
            p=p,
            matrix_size_err=args.matrix_size_err,
            matrix_size_time=args.matrix_size_time,
            n_trials=args.n_trials,
            vary_condition=args.other_distrib,
        )

    # Setup the single horizontal legend at the bottom
    handles, labels = axes[0, 1].get_legend_handles_labels()

    # INCREASED SIZE HERE: added fontsize=12, markerscale=1.5, and adjusted layout
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,  # Switched to 2 columns so the larger text doesn't overlap
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=14,
        markerscale=1.5,
    )

    # INCREASED MARGIN HERE: rect bottom changed from 0.08 to 0.12 to give the bigger legend space
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
