"""
Layer-wise dynamic Remez (SLSQP minimax) coefficient computation.

Solves for optimal polynomial coefficients (a, b, c) layer-by-layer to approximate
fractional powers. To avoid non-linear parameter space traps in deep compositions,
the target exponent p is decomposed as q = p^(1/N). Each layer approximates x^q
over the distorted domain mapped by the previous layer.

The iteration:  x_{n+1} = a_n·x_n + b_n·x_n³ + c_n·x_n⁵
Goal:           after N steps, x_N ≈ x_0^p   for x_0 ∈ [eps, 1]
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Optional

from smuon.coeffs.polar_express import optimal_composition


def solve_fixed(
    num_steps: int,
    p_order: float,
    eps: float = 0.005,
    cushion: float = 0.02,
    safety_factor: float = 1.005,
    p_threshold: float = 30.0,
) -> Optional[List[Tuple[float, float, float]]]:
    """
    Compute layer-wise optimal coefficients for Newton-Schulz iterations.

    Parameters
    ----------
    num_steps : int
        Number of iterations N.
    p_order : float
        Target exponent p ∈ (0, 1). After N steps, x_N ≈ x_0^p.
    eps: float
        Lower bound of the operational domain.
    cushion: float
        Artificial lower bound lift to ease optimization in early steps.
    safety_factor: float
        Under-approximation scaler to prevent catastrophic upper-bound overflow.
    """
    if p_order <= 0 or p_order == float("inf"):
        return [(1.0, 0.0, 0.0)]
    if abs(p_order - 1.0) <= 2e-2:
        return None
    if p_order <= (1 / p_threshold):
        return optimal_composition(
            l=eps, num_iters=5, safety_factor_eps=1e-2, cushion=0.0
        )

    # Decompose the target power equally across N layers
    q = p_order ** (1.0 / num_steps)

    # Initial domain bounds
    a_bound, b_bound = eps, 1.0
    coeffs = []
    num_nodes = 500

    for step in range(num_steps):
        # TRICK 1: Domain Cushioning
        # Lift the lower bound during optimization to avoid infinite slopes near zero
        opt_a_bound = max(a_bound, cushion * b_bound)

        # Chebyshev nodes mapped to the cushioned domain
        nodes = np.cos(np.pi * (2 * np.arange(num_nodes) + 1) / (2 * num_nodes))
        x_grid = 0.5 * (b_bound - opt_a_bound) * nodes + 0.5 * (b_bound + opt_a_bound)
        y_target = x_grid**q

        def objective(vars):
            return vars[-1]

        def constraint_upper(vars):
            c1, c3, c5, t = vars
            return t - ((c1 * x_grid + c3 * x_grid**3 + c5 * x_grid**5) - y_target)

        def constraint_lower(vars):
            c1, c3, c5, t = vars
            return t + ((c1 * x_grid + c3 * x_grid**3 + c5 * x_grid**5) - y_target)

        # TRICK 2: Endpoint Pinning
        # Force the polynomial to pass exactly through (1, 1).
        # Prevents compounding drift where x > 1 causes divergence.
        def constraint_pin_endpoint(vars):
            c1, c3, c5, t = vars
            return (c1 + c3 + c5) - 1.0  # Must equal 0

        x0 = [1.0, 0.0, 0.0, 0.1]
        bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (0.0, None)]
        cons = [
            {"type": "ineq", "fun": constraint_upper},
            {"type": "ineq", "fun": constraint_lower},
            {"type": "eq", "fun": constraint_pin_endpoint},
        ]

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
        )

        if not res.success:
            # Fallback to identity if optimization fails to preserve stability
            c1, c3, c5 = 1.0, 0.0, 0.0
        else:
            c1, c3, c5 = res.x[0], res.x[1], res.x[2]

            # TRICK 3: Safety Factor
            # Slightly under-approximate to guarantee we don't accidentally exceed the upper bound.
            if step < num_steps - 1:  # Don't scale the final output layer
                c1 /= safety_factor
                c3 /= safety_factor**3
                c5 /= safety_factor**5

        coeffs.append((float(c1), float(c3), float(c5)))

        # Determine the exact distorted domain for the next layer
        # Note: We evaluate mapping on the TRUE a_bound, not the cushioned one
        mapped_nodes = np.cos(np.pi * (2 * np.arange(num_nodes) + 1) / (2 * num_nodes))
        mapped_grid_base = 0.5 * (b_bound - a_bound) * mapped_nodes + 0.5 * (
            b_bound + a_bound
        )
        mapped_grid = (
            c1 * mapped_grid_base + c3 * mapped_grid_base**3 + c5 * mapped_grid_base**5
        )

        a_bound = np.min(mapped_grid)
        b_bound = np.max(mapped_grid)

        # Safety check to prevent total domain collapse
        if a_bound < 0 or b_bound > 10.0:
            a_bound, b_bound = eps, 1.0

    return coeffs


def solve_adaptive(
    p_order: float,
    tol: float = 2e-1,
    max_steps: int = 10,
    eps: float = 0.005,
    p_threshold: float = 30.0,
) -> List[Tuple[float, float, float]]:
    """
    Error-driven solver: finds the minimum number of iterations (up to max_steps)
    required to achieve the target L_inf (Maximum Absolute Error) tolerance.
    """
    if p_order <= 0 or p_order == float("inf"):
        best_coeffs = [(1.0, 0.0, 0.0)]
    elif abs(p_order - 1.0) <= 2e-2:
        best_coeffs = None
    elif p_order <= (1 / p_threshold):
        best_coeffs = optimal_composition(
            l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.03
        )
    else:
        # Evaluate over the true domain [eps, 1] using Chebyshev nodes
        num_nodes = 1000
        nodes = np.cos(np.pi * (2 * np.arange(num_nodes) + 1) / (2 * num_nodes))
        x_grid = 0.5 * (1.0 - eps) * nodes + 0.5 * (1.0 + eps)
        y_target = x_grid**p_order

        best_coeffs = None
        best_error = float("inf")

        for n in range(1, max_steps + 1):
            # Pass the unified eps down to the static solver
            coeffs = solve_fixed(n, p_order, eps=eps)

            if not coeffs:
                continue

            # Roll out the composition
            y_pred = x_grid.copy()
            for a, b, c in coeffs:
                y_pred = a * y_pred + b * y_pred**3 + c * y_pred**5

            error = np.max(np.abs(y_pred - y_target))

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
