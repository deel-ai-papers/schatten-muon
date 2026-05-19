import torch
import numpy as np
from smuon.svs.base import PApproximator
from smuon.svs.approx import (
    svd_lowrank_and_residual,
    topk_svs_from_gram,
    _evaluate_bounded_log_J,
)
from smuon.svs.common import maximize_p


class ApproxPApproximator(PApproximator):
    """
    Approximate p* via subsampled SVD with bounded tail correction.

    State variables (flat keys for backward compatibility):
    - grad_svs_ema : Tensor - EMA of top-k gradient singular values
    - act_svs_ema : Tensor - EMA of top-k activation singular values
    - R_G_ema : float - EMA of gradient residual spectral norm
    - R_A_ema : float - EMA of activation residual spectral norm
    - E_G_ema : float - EMA of gradient residual energy
    - E_A_ema : float - EMA of activation residual energy

    Complexity: O(mn·k) where k = subsampling_ratio * min(m,n).
    """

    STATE_NAMESPACE = None  # Use flat keys for backward compatibility

    def create_state(self, state):
        pass  # Lazily initialized

    def update_and_compute_p(self, state, grad_2d, act_2d, use_gram, **kwargs):
        # --- Gradient: randomized SVD (always a raw matrix, not Gram) ---
        k_G = max(1, int(self.subsampling_ratio * min(grad_2d.shape)))
        k_G = min(k_G, min(grad_2d.shape) - 1)

        S_G, R_G = svd_lowrank_and_residual(grad_2d.float(), k_G)
        frob_G_sq = grad_2d.float().pow(2).sum().item()
        E_G = max(0.0, frob_G_sq - np.sum(S_G**2))
        d_G = min(grad_2d.shape) - k_G

        # --- Activations: branch on Gram vs raw ---
        if use_gram:
            # Gram path: use eigendecomposition with spectral-norm
            # grounding (σ_k bound). Avoids the noisy random-projection
            # residual estimate that causes R_A^(q_A-2) overflow.
            k_A = max(1, int(self.subsampling_ratio * act_2d.shape[0]))
            k_A = min(k_A, act_2d.shape[0] - 1)
            S_A, R_A, E_A, d_A = topk_svs_from_gram(act_2d, k_A)
        else:
            k_A = max(1, int(self.subsampling_ratio * min(act_2d.shape)))
            k_A = min(k_A, min(act_2d.shape) - 1)
            S_A, R_A = svd_lowrank_and_residual(act_2d.float(), k_A)
            frob_A_sq = act_2d.float().pow(2).sum().item()
            E_A = max(0.0, frob_A_sq - np.sum(S_A**2))
            d_A = min(act_2d.shape) - k_A

        # Convert to tensors for EMA
        S_G_t = torch.from_numpy(S_G)
        S_A_t = torch.from_numpy(S_A)

        # Apply EMA to top-k singular values and residual stats
        if "grad_svs_ema" not in state:
            state["grad_svs_ema"] = S_G_t.clone()
            state["act_svs_ema"] = S_A_t.clone()
            state["R_G_ema"] = R_G
            state["R_A_ema"] = R_A
            state["E_G_ema"] = E_G
            state["E_A_ema"] = E_A
        else:
            if state["grad_svs_ema"].shape == S_G_t.shape:
                state["grad_svs_ema"].lerp_(S_G_t, 1 - self.sv_momentum)
            else:
                state["grad_svs_ema"] = S_G_t.clone()
            if state["act_svs_ema"].shape == S_A_t.shape:
                state["act_svs_ema"].lerp_(S_A_t, 1 - self.sv_momentum)
            else:
                state["act_svs_ema"] = S_A_t.clone()
            # EMA for scalar residual stats
            state["R_G_ema"] = (
                self.sv_momentum * state["R_G_ema"] + (1 - self.sv_momentum) * R_G
            )
            state["R_A_ema"] = (
                self.sv_momentum * state["R_A_ema"] + (1 - self.sv_momentum) * R_A
            )
            state["E_G_ema"] = (
                self.sv_momentum * state["E_G_ema"] + (1 - self.sv_momentum) * E_G
            )
            state["E_A_ema"] = (
                self.sv_momentum * state["E_A_ema"] + (1 - self.sv_momentum) * E_A
            )

        # Use EMA'd values for optimization
        S_G_ema = state["grad_svs_ema"].numpy()
        S_A_ema = state["act_svs_ema"].numpy()

        def neg_log_J(p):
            return -_evaluate_bounded_log_J(
                p,
                S_G_ema,
                S_A_ema,
                state["R_G_ema"],
                state["R_A_ema"],
                state["E_G_ema"],
                state["E_A_ema"],
                d_G,
                d_A,
            )

        p_star, _ = maximize_p(neg_log_J, bounds=(self.pmin, self.pmax))
        return p_star
