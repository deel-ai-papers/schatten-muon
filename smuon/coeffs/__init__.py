"""
Methods for computing x^p via polynomial iteration.

Each method exposes a single `solve(num_steps, p_order)` function that
returns a list of (a, b, c, d) coefficient tuples — one per step.

    from methods import fixed_lp, fixed_remez, varying_per_step

    coeffs = fixed_lp.solve(num_steps=6, p_order=0.5)
    # coeffs is a list of 6 identical np.ndarray([a, b, c, d])

    coeffs = varying_per_step.solve(num_steps=6, p_order=0.5)
    # coeffs is a list of 6 different np.ndarray([a, b, c, d])
"""

from smuon.coeffs.utils import apply_step, compose
from smuon.coeffs import l2_lagrangian, linf_remez

# Default tolerance values for adaptive coefficient methods
_DEFAULT_L2_TOL = 1e-2
_DEFAULT_LINF_TOL = 2e-1

COEFF_METHODS = {
    "l2_lagrangian_static": l2_lagrangian.solve_fixed,
    "linf_remez_static": linf_remez.solve_fixed,
    "l2_lagrangian_adaptive": l2_lagrangian.solve_adaptive,
    "linf_remez_adaptive": linf_remez.solve_adaptive,
}
