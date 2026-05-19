from abc import ABC, abstractmethod


class SecondOrderMoment(ABC):
    """
    Base class for second-order moment estimation (preconditioning).

    Attributes
    ----------
    p : float or None
        Schatten-p order. When set, scales the power exponent as -1/(4p).
        When None, uses fixed exponent -0.25.
    beta2 : float
        Decay rate for exponential moving average (ignored by cumulative variants).
    eps : float
        Numerical stability constant.
    use_bias_correction : bool
        Whether to apply bias correction for EMA-based moments.
    """

    # Subclasses should set this to True if they use EMA (not cumulative sum)
    USES_EMA = True

    def __init__(
        self, p=None, beta2=0.999, eps=1e-8, p_threshold=30.0, use_bias_correction=True
    ):
        self.p = p
        self.beta2 = beta2
        self.eps = eps
        self.p_threshold = p_threshold
        self.use_bias_correction = use_bias_correction

    @property
    def uses_p(self):
        """Whether this moment scales with Schatten-p order."""
        return self.p is not None

    def _get_exponent(self):
        """Get the power exponent based on p setting."""
        if self.p is not None:
            return -1.0 / (2.0 * (self.p + 1))
        return -0.25

    def _get_bias_correction(self, state):
        """
        Get bias correction factor for EMA-based moments.

        Returns 1.0 if bias correction is disabled or not applicable.
        """
        if not self.use_bias_correction or not self.USES_EMA:
            return 1.0
        step = state.get("moment_step", 1)
        return 1.0 / (1.0 - self.beta2**step)

    def _increment_step(self, state):
        """Increment the step counter for bias correction."""
        if "moment_step" not in state:
            state["moment_step"] = 0
        state["moment_step"] += 1

    @abstractmethod
    def update(self, state, grad):
        """
        Update moment state and return preconditioning matrix.

        Parameters
        ----------
        state : dict
            Optimizer state dictionary for this parameter.
        grad : Tensor
            Gradient tensor.

        Returns
        -------
        Tensor
            Preconditioning matrix D_t (same shape as 2D grad).
        """
        pass

    def __call__(self, state, grad):
        """Alias for update()."""
        return self.update(state, grad)
