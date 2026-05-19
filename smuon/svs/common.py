"""
Shared utilities for optimal p* computation.

All methods ultimately need to evaluate:

    J(p) = ||G||_{1+1/p} / ||A||_{2(p+1)/(p-1)}

in log-space given estimates of the Schatten-norm power sums.
This module provides the shared optimization wrapper (grid or continuous).
"""

import torch
from scipy.optimize import minimize_scalar


def log_J_from_log_schatten_sums(p, log_sum_g, log_sum_a):
    """
    Compute log J(p) given:
        log_sum_g = log(Σ σ_i(G)^{1+1/p})
        log_sum_a = log(Σ σ_j(A)^{2(p+1)/(p-1)})

    Returns log J(p) = log ||G||_{q*} - log ||A||_{k(p)}
    """
    log_num = (p / (p + 1.0)) * log_sum_g
    log_den = ((p - 1.0) / (2.0 * (p + 1.0))) * log_sum_a
    return log_num - log_den


def maximize_p(neg_log_J_fn, grid=None, bounds=(1.05, 100.0)):
    """
    Maximize J(p) over p >= 1.

    Parameters
    ----------
    neg_log_J_fn : callable
        Function p -> -log J(p)  (scalar in, scalar out).
    grid : list[float] or None
        If provided, evaluate on this grid and return the best p.
        If None, use scipy bounded optimization on `bounds`.
    bounds : tuple
        (lo, hi) for the continuous optimizer. Ignored if grid is set.

    Returns
    -------
    p_star : float
    log_J_star : float
        The value log J(p*) at the optimum.
    """
    if grid is not None:
        best_p, best_val = grid[0], -neg_log_J_fn(grid[0])
        for p in grid[1:]:
            val = -neg_log_J_fn(p)
            if val > best_val:
                best_val, best_p = val, p
        return best_p, best_val
    else:
        res = minimize_scalar(neg_log_J_fn, bounds=bounds, method="bounded")
        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")
        return res.x, -res.fun
