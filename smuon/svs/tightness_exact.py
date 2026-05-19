""" 
Tightness-corrected p* with preconditioning-aware criterion.

State (under STATE_NAMESPACE="sv_state"):
    C_diag_ema : EMA of (U_tilde^T G_tilde V_tilde) diagonal
    B_diag_ema : EMA of diag(V_tilde^T A A^T V_tilde)
"""

import torch
from scipy.optimize import minimize_scalar
from smuon.svs.base import PApproximator


def _promote_common(*tensors, min_dtype=None):
    dtypes = [t.dtype for t in tensors if t is not None]
    if not dtypes:
        return tensors
    target = dtypes[0]
    for d in dtypes[1:]:
        target = torch.promote_types(target, d)
    if min_dtype is not None:
        target = torch.promote_types(target, min_dtype)
    return tuple(t.to(target) if t is not None else None for t in tensors)


def _orient_activation(V_M, act_tensor, use_gram):
    n_expected = V_M.shape[-2]
    shape = tuple(act_tensor.shape)

    if use_gram:
        if act_tensor.dim() == 2 and shape == (n_expected, n_expected):
            return act_tensor
        raise ValueError(
            f"use_gram=True: activation must be a 2D Gram matrix of shape "
            f"({n_expected}, {n_expected}), got {shape}."
        )

    if act_tensor.dim() == 2:
        if shape[0] == n_expected:
            return act_tensor
        if shape[1] == n_expected:
            return act_tensor.transpose(-2, -1)

    if act_tensor.dim() >= 3 and shape[-1] == n_expected:
        return act_tensor.reshape(-1, n_expected).transpose(-2, -1)

    if act_tensor.dim() >= 3 and shape[0] == n_expected:
        return act_tensor.reshape(n_expected, -1)

    raise ValueError(
        f"Activation shape {shape} has no axis matching the layer's input "
        f"feature dim {n_expected}."
    )


def _compute_ND_exact(p, C_diag, B_diag, S_M):
    """
    N(p) = sum_i  sigma^{1/p}  * C_i
    D(p) = sum_i  sigma^{2/p}  * B_i
    """
    p_safe = min(max(p, 1.0 + 1e-9), 1.0e4)
    inv_p = 1.0 / p_safe
    two_inv_p = 2.0 / p_safe
    N = torch.sum(C_diag * (S_M**inv_p))
    D = torch.sum(B_diag * (S_M**two_inv_p))
    return N, D


class ExactTightnessPApproximator(PApproximator):
    """
    Tightness-corrected p*, optionally with a preconditioning-aware objective.

    Usage
    -----
    approximator.update_and_compute_p(state, G, A, use_gram=False,
                                          nesterov=True, mom_2d=M, beta1=b1)

    """

    STATE_NAMESPACE = "sv_state"

    def create_state(self, state):
        pass

    def update_and_compute_p(
        self,
        state,
        grad_2d,
        act_2d,
        use_gram,
        nesterov=False,
        mom_2d=None,
        beta1=None,
        **kwargs,
    ):
        if mom_2d is None or beta1 is None:
            raise ValueError("ExactTightnessPApproximator requires mom_2d and beta1")

        grad_2d, act_2d, mom_2d = _promote_common(
            grad_2d, act_2d, mom_2d, min_dtype=torch.float32
        )

        # Align persisted EMA state if it was created in a different dtype.
        if self.STATE_NAMESPACE in state:
            svs = state[self.STATE_NAMESPACE]
            target = grad_2d.dtype
            for k, v in list(svs.items()):
                if isinstance(v, torch.Tensor) and v.dtype != target:
                    svs[k] = v.to(target)

        # Expected momentum M_t (out-of-place; optionally Nesterov-corrected)
        M_t = mom_2d.lerp(grad_2d, 1 - beta1)
        M_expected = M_t.lerp(grad_2d, 1 - beta1) if nesterov else M_t

        M_for_svd = M_expected
        G_for_C = grad_2d

        # Full SVD
        U_M, S_M, Vh_M = torch.linalg.svd(M_for_svd, full_matrices=False)
        V_M = Vh_M.transpose(-2, -1)

        # Alignment coefficients C_ii = (U^T G_for_C V)_ii
        C_diag = torch.einsum("mi,mn,ni->i", U_M, G_for_C, V_M)

        A_oriented = _orient_activation(V_M, act_2d, use_gram)
        if use_gram:
            VT_GA = V_M.transpose(-2, -1) @ A_oriented
            B_diag = torch.sum(VT_GA * V_M.transpose(-2, -1), dim=-1)
        else:
            VT_A = V_M.transpose(-2, -1) @ A_oriented
            B_diag = torch.sum(VT_A * VT_A, dim=-1)
        B_diag = torch.clamp(B_diag, min=0.0)

        # EMA on noisy quantities only. S_M is not smoothed — momentum is
        # itself an EMA upstream.
        if self.STATE_NAMESPACE not in state:
            state[self.STATE_NAMESPACE] = {
                "C_diag_ema": C_diag.clone().detach(),
                "B_diag_ema": B_diag.clone().detach(),
            }
        else:
            svs = state[self.STATE_NAMESPACE]
            if svs["C_diag_ema"].shape == C_diag.shape:
                svs["C_diag_ema"].lerp_(C_diag, 1 - self.sv_momentum)
            else:
                svs["C_diag_ema"] = C_diag.clone().detach()
            if svs["B_diag_ema"].shape == B_diag.shape:
                svs["B_diag_ema"].lerp_(B_diag, 1 - self.sv_momentum)
            else:
                svs["B_diag_ema"] = B_diag.clone().detach()

        C_s = state[self.STATE_NAMESPACE]["C_diag_ema"]
        B_s = state[self.STATE_NAMESPACE]["B_diag_ema"]

        # Bounded 1D search over p for J(p) = N(p)^2 / D(p)
        def objective(p):
            N, D = _compute_ND_exact(p, C_s, B_s, S_M)
            return -((N * N) / (D + 1e-12)).item()

        res = minimize_scalar(
            objective, bounds=(self.pmin, self.pmax), method="bounded"
        )
        p_star = float(res.x)

        # alpha_star = eta*(p_star) / eta*(inf).
        # At p->inf, sigma^{1/p} -> 1 and eta*(inf) = sum(C) / (2 sum(B)).
        N_p, D_p = _compute_ND_exact(p_star, C_s, B_s, S_M)
        N_inf, D_inf = _compute_ND_exact(1.0e8, C_s, B_s, S_M)

        eta_p_val = (N_p / (2.0 * D_p + 1e-12)).item()
        eta_inf_val = (N_inf / (2.0 * D_inf + 1e-12)).item()

        if eta_inf_val <= 0 or eta_p_val <= 0:
            alpha_star = 1.0
        else:
            alpha_star = eta_p_val / eta_inf_val

        # The quadratic model says alpha_star >= 1 by construction at p = p*
        # (p* maximizes decrease). Numerically it can come out slightly <1;
        # clip to a sane range to prevent runaway LR.
        alpha_star = max(0.5, min(alpha_star, 2.0))

        return p_star, float(alpha_star)
