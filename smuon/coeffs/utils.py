"""
Shared utilities for polynomial iteration methods.

The iteration:  x_{n+1} = a·x_n + b·x_n³ + c·x_n⁵
Goal:           after N steps, x_N ≈ x_0^p   for x_0 ∈ (0, 1]

Only odd powers are used so that the iteration generalises to matrices:
    X_{n+1} = a·X + b·X(XᵀX) + c·X(XᵀX)²
which preserves the shape of any rectangular X.
"""

import numpy as np


def basis(x):
    """Odd polynomial basis: [x, x³, x⁵]."""
    x2 = x**2
    return np.column_stack([x, x * x2, x * x2**2])


def apply_step(x, coeffs):
    """Apply one iteration: x_{n+1} = a·x + b·x³ + c·x⁵."""
    a, b, c = coeffs
    x2 = x**2
    return a * x + b * x * x2 + c * x * x2**2


def compose(x0, coeffs_list):
    """
    Apply a sequence of iterations and return the full trajectory.

    Parameters
    ----------
    x0 : np.ndarray
        Starting values.
    coeffs_list : list of (a, b, c) tuples/arrays
        Coefficients for each step.

    Returns
    -------
    trajectory : list of np.ndarray
        [x0, x1, x2, ..., xN]
    """
    trajectory = [x0.copy()]
    x = x0.copy()
    for coeffs in coeffs_list:
        x = apply_step(x, coeffs)
        x = np.clip(x, 0, None)
        trajectory.append(x.copy())
    return trajectory
