from smuon.svs.grad_exact import ExactPApproximator
from smuon.svs.grad_approx import ApproxPApproximator
from smuon.svs.momentum_approx import MomentumApproxPApproximator
from smuon.svs.momentum_exact import ExactMomentumPApproximator
from smuon.svs.tightness_exact import ExactTightnessPApproximator
from smuon.svs.tightness_approx import ApproxTightnessPApproximator


# Registry mapping p_method names to approximator classes
P_APPROXIMATOR_REGISTRY = {
    "exact": ExactPApproximator,
    "approx": ApproxPApproximator,
    "approx_momentum": MomentumApproxPApproximator,
    "exact_momentum": ExactMomentumPApproximator,
    "exact_tightness": ExactTightnessPApproximator,
    "approx_tightness": ApproxTightnessPApproximator,
}


def create_p_approximator(
    method, pmin=1.02, pmax=35.0, subsampling_ratio=0.1, sv_momentum=0.95
):
    """
    Factory function to create a P-approximator instance.

    Parameters
    ----------
    method : str
        Approximation method: "exact", "approx", "approx_momentum", or "exact_momentum".
    pmin, pmax : float
        Bounds for p* optimization.
    subsampling_ratio : float
        Fraction of singular values to compute (ignored by exact).
    sv_momentum : float
        EMA decay rate for singular value smoothing.

    Returns
    -------
    PApproximator
        Configured approximator instance.

    Raises
    ------
    ValueError
        If method is not in P_APPROXIMATOR_REGISTRY.
    """
    if method not in P_APPROXIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown p_method '{method}'. "
            f"Choose from: {list(P_APPROXIMATOR_REGISTRY.keys())}"
        )
    cls = P_APPROXIMATOR_REGISTRY[method]
    return cls(
        pmin=pmin,
        pmax=pmax,
        subsampling_ratio=subsampling_ratio,
        sv_momentum=sv_momentum,
    )
