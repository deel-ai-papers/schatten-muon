"""
Safe Lagrangian coefficient computation (Epsilon-Aware).

Solves for optimal polynomial coefficients (a, b, c) for a given target exponent p
using constrained least squares over the domain [eps, 1].

The iteration:  x_{n+1} = a·x + b·x³ + c·x⁵
Goal:           after N steps, x_N ≈ x_0^p   for x_0 ∈ [eps, 1]
"""

import torch
from typing import Tuple, List

from smuon.coeffs.polar_express import optimal_composition


def _solve_coeffs(
    p: float, eps: float = 0.0, p_threshold: float = 30.0
) -> Tuple[float, float, float]:
    """
    Solve for optimal coefficients by minimizing the L2 integral from eps to 1.

    The integral of x^n from eps to 1 is (1 - eps^(n+1)) / (n+1).
    """
    if abs(p - 1.0) <= 1e-2:
        return 1.0, 0.0, 0.0
    if p >= p_threshold:
        res = optimal_composition(
            l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.03
        )
        return res[0] if isinstance(res, list) else res

    # Helper to compute the definite integral on [eps, 1]
    def I(n):
        return (1.0 - eps ** (n + 1)) / (n + 1)

    # Construct Linear System A * [a, b, c, lambda] = B
    # Entries in A correspond to partial derivatives of the integral sum.
    A = torch.tensor(
        [
            [I(2), I(4), I(6), 1.0],
            [I(4), I(6), I(8), 1.0],
            [I(6), I(8), I(10), 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    B = torch.tensor([I(p + 1), I(p + 3), I(p + 5), 1.0], dtype=torch.float64)

    x = torch.linalg.solve(A, B)
    return x[0].item(), x[1].item(), x[2].item()


def solve_fixed(
    num_steps: int, p_order: float, eps: float = 0.001, p_threshold: float = 30.0
) -> List[Tuple[float, float, float]]:
    """Fixed-step solver using eps-aware integrals."""
    if p_order <= 0 or p_order == float("inf"):
        return [(1.0, 0.0, 0.0)] * num_steps
    elif abs(p_order - 1.0) <= 2e-2:
        return None
    elif p_order <= (1 / p_threshold):
        return optimal_composition(
            l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.03
        )
    else:
        q = p_order ** (1.0 / num_steps)
        coeffs = _solve_coeffs(q, eps=eps)
        return [coeffs] * num_steps


def solve_adaptive(
    p_order: float,
    tol: float = 2e-2,
    max_steps: int = 10,
    eps: float = 0.001,
    p_threshold: float = 30.0,
) -> List[Tuple[float, float, float]]:
    """Adaptive solver that passes eps down to the integral solver."""
    if p_order <= 0 or p_order == float("inf"):
        best_coeffs = [(1.0, 0.0, 0.0)]
    elif abs(p_order - 1.0) <= 2e-2:
        best_coeffs = None
    elif p_order <= (1 / p_threshold):
        best_coeffs = optimal_composition(
            l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.03
        )
    else:
        x_grid = torch.linspace(eps, 1.0, 1000, dtype=torch.float64)
        y_target = x_grid**p_order

        best_coeffs = None
        best_error = float("inf")

        for n in range(1, max_steps + 1):
            # We solve for the coefficients specifically for this eps
            coeffs = solve_fixed(n, p_order, eps=eps)

            y_pred = x_grid.clone()
            for a, b, c in coeffs:
                y_pred = a * y_pred + b * y_pred**3 + c * y_pred**5

            error = torch.mean((y_pred - y_target) ** 2).item()

            if error < best_error:
                best_error = error
                best_coeffs = coeffs

            if error <= tol:
                break

            if best_coeffs is None:
                best_coeffs = [(1.0, 0.0, 0.0)]

    # Pad with identity, the compiler will handle the rest
    while len(best_coeffs) < max_steps:
        best_coeffs.append((1.0, 0.0, 0.0))

    return best_coeffs
