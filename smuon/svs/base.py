from abc import ABC, abstractmethod


class PApproximator(ABC):
    """
    Base class for optimal p* approximation methods.

    Computes p* = argmax J(p) where J(p) = ||G||_{q*}^{q*} / ||A||_{k(p)}^{k(p)}
    using different singular value approximation strategies.

    Attributes
    ----------
    pmin, pmax : float
        Bounds for p* optimization.
    subsampling_ratio : float
        Fraction of singular values to compute (ignored by exact methods).
    sv_momentum : float
        EMA decay for singular value tracking (beta).
    """

    # Subclasses should set this to identify their state variables
    # None means flat keys for backward compatibility
    STATE_NAMESPACE = None

    def __init__(self, pmin=1.02, pmax=35.0, subsampling_ratio=0.1, sv_momentum=0.95):
        self.pmin = pmin
        self.pmax = pmax
        self.subsampling_ratio = subsampling_ratio
        self.sv_momentum = sv_momentum

    @abstractmethod
    def create_state(self, state):
        """
        Initialize method-specific state variables.

        Parameters
        ----------
        state : dict
            Optimizer state dictionary for this parameter.

        Notes
        -----
        State should be stored under self.STATE_NAMESPACE to avoid collisions.
        For backward compatibility, exact/approx use flat keys (legacy format).
        """
        pass

    @abstractmethod
    def update_and_compute_p(self, state, grad_2d, act_2d, use_gram, **kwargs):
        """
        Update state and compute optimal p*.

        Parameters
        ----------
        state : dict
            Optimizer state dictionary for this parameter.
        grad_2d : Tensor (m, n)
            2D gradient tensor.
        act_2d : Tensor (m', n') or (d, d)
            2D activation tensor, or Gram matrix if use_gram=True.
        use_gram : bool
            If True, act_2d is a Gram matrix (A^T A) that has been
            all-reduced across GPUs. Singular values extracted via eigvalsh.
        **kwargs : dict
            Method-specific arguments (e.g., momentum buffer for approx_momentum).

        Returns
        -------
        p_star : float or None
            Optimal Schatten norm order, or None if computation failed.

        Notes
        -----
        - This method should handle NaN detection and return None if computation fails
        - EMA updates should be applied internally before optimization
        - Distributed training is handled via use_gram flag
        """
        pass

    def get_state_for_checkpoint(self, state):
        """
        Extract serializable state for checkpointing.

        Returns a dict that can be pickled and loaded later.
        Override if your state contains non-serializable objects.
        """
        if self.STATE_NAMESPACE is not None:
            return state.get(self.STATE_NAMESPACE, {})
        return {}  # Legacy flat-key methods handle this in the optimizer
