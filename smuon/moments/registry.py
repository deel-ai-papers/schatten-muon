from smuon.moments.adam import AdamMoment
from smuon.moments.adagrad import AdaGradMoment
from smuon.moments.adafactor import AdafactorMoment
from smuon.moments.sania import SaniaMoment


# Registry mapping names to (class, parametric) tuples
# parametric=True means the moment scales with p, False means fixed -0.25 exponent
MOMENT_REGISTRY = {
    "adam": (AdamMoment, False),
    "adagrad": (AdaGradMoment, False),
    "adafactor": (AdafactorMoment, False),
    "sania": (SaniaMoment, False),
    "padam": (AdamMoment, True),
    "padagrad": (AdaGradMoment, True),
    "padafactor": (AdafactorMoment, True),
    "psania": (SaniaMoment, True),
}

# For backwards compatibility
MOMENT_CLASSES = {k: v[0] for k, v in MOMENT_REGISTRY.items()}


def create_moment(
    name,
    p=None,
    beta2=0.999,
    eps=1e-8,
    use_bias_correction=True,
    p_threshold=float("inf"),
):
    """
    Factory function to create a moment instance.

    Parameters
    ----------
    name : str
        Moment type: "adam", "adagrad", "adafactor", "sania", "padam", "padagrad",
        "padafactor", "psania", or "none".
        Parametric variants (padam, padagrad, etc.) scale exponent with p.
        Non-parametric variants use fixed exponent (-0.25 for most, -0.5 for sania).
    p : float or None
        Schatten-p order for parametric scaling (only used by parametric variants).
    beta2 : float
        EMA decay rate.
    eps : float
        Numerical stability constant.
    use_bias_correction : bool
        Whether to apply bias correction for EMA-based moments.

    Returns
    -------
    SecondOrderMoment or None
        Moment instance, or None if name is "none".
    """
    if name == "none":
        return None
    if name not in MOMENT_REGISTRY:
        raise ValueError(
            f"Unknown moment type '{name}'. "
            f"Choose from: {list(MOMENT_REGISTRY.keys()) + ['none']}"
        )
    cls, parametric = MOMENT_REGISTRY[name]
    # Only pass p for parametric variants
    return cls(
        p=p if parametric else None,
        beta2=beta2,
        eps=eps,
        use_bias_correction=use_bias_correction,
        p_threshold=p_threshold,
    )


def is_parametric(name):
    """Check if a moment type scales with p."""
    if name == "none" or name not in MOMENT_REGISTRY:
        return False
    return MOMENT_REGISTRY[name][1]
