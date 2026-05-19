import torch
import torch.distributed as dist
from smuon.svs.base import PApproximator
from smuon.svs.exact import (
    get_full_svs,
    get_svs_from_gram,
    optimal_p as exact_optimal_p,
)


class ExactPApproximator(PApproximator):
    """
    Exact p* computation via full SVD with EMA smoothing.

    State variables (flat keys for backward compatibility):
    - grad_svs_ema : Tensor - EMA of gradient singular values
    - act_svs_ema : Tensor - EMA of activation singular values

    Complexity: O(mn·min(m,n)) per parameter per update.
    """

    STATE_NAMESPACE = None  # Use flat keys for backward compatibility

    def create_state(self, state):
        # State is lazily initialized in update_and_compute_p
        # This avoids allocating tensors before knowing their shape
        pass

    def update_and_compute_p(self, state, grad_2d, act_2d, use_gram, **kwargs):
        # Compute full SVs
        grad_svs = get_full_svs(grad_2d.float())

        if use_gram:
            # act_2d is already the (reduced) Gram matrix
            act_svs = get_svs_from_gram(act_2d)
        else:
            act_svs = get_full_svs(act_2d.float())

        if grad_svs is None or act_svs is None:
            return None

        grad_svs = grad_svs.cpu()
        act_svs = act_svs.cpu()

        # Initialize or update EMA
        if "grad_svs_ema" not in state:
            state["grad_svs_ema"] = grad_svs.clone()
            state["act_svs_ema"] = act_svs.clone()
        else:
            # Handle dimension changes gracefully
            if state["grad_svs_ema"].shape == grad_svs.shape:
                state["grad_svs_ema"].lerp_(grad_svs, 1 - self.sv_momentum)
            else:
                state["grad_svs_ema"] = grad_svs.clone()

            if state["act_svs_ema"].shape == act_svs.shape:
                state["act_svs_ema"].lerp_(act_svs, 1 - self.sv_momentum)
            else:
                state["act_svs_ema"] = act_svs.clone()

        # Check distributed training requirements
        if dist.is_initialized() and not use_gram:
            raise RuntimeError(
                "Distributed training requires use_gram=True for correct "
                "singular value aggregation. Averaging singular values "
                "across ranks is mathematically invalid; use Gram matrices "
                "which can be correctly summed before eigendecomposition."
            )

        # Compute p* using EMA'd singular values
        p_star, _ = exact_optimal_p(
            state["grad_svs_ema"],
            state["act_svs_ema"],
            bounds=(self.pmin, self.pmax),
        )

        return p_star
