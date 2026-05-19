import torch
from scipy.optimize import minimize_scalar
from smuon.svs.base import PApproximator


def _promote_common(*tensors, min_dtype=None):
    """
    Promote tensors to a common dtype via torch.promote_types.

    If min_dtype is given, the result is additionally promoted to at least
    that dtype. Used by SVD-bearing code paths to enforce fp32 regardless
    of input precision, since LAPACK/MAGMA SVD backends don't support bf16.
    Stays in the caller's dtype when it's already wide enough (e.g. fp32
    inputs don't get re-cast to fp32; fp64 inputs stay fp64).
    """
    dtypes = [t.dtype for t in tensors if t is not None]
    if not dtypes:
        return tensors
    target = dtypes[0]
    for d in dtypes[1:]:
        target = torch.promote_types(target, d)
    if min_dtype is not None:
        target = torch.promote_types(target, min_dtype)
    return tuple(t.to(target) if t is not None else None for t in tensors)


def evaluate_fp(p, C_diag, sigma_M, norm_A_Skp):
    """
    Evaluate the momentum-aligned objective function f_p.
    """
    p_safe = min(p, 10000.0)
    num = (torch.sum(C_diag * (sigma_M ** (1.0 / p_safe)))) ** 2
    term2 = torch.sum(sigma_M ** ((p_safe + 1.0) / p_safe)) ** (2.0 / (p_safe + 1.0))
    den = (norm_A_Skp**2) * term2
    return num / (den + 1e-12)


class ExactMomentumPApproximator(PApproximator):
    """
    Exact momentum-aligned p* computation via full SVD.

    Key insight: Momentum M_t is naturally smooth (high beta1 ≈ 0.95), so we use
    its instantaneous SVs without EMA. Activations and alignment (grad·mom) are
    noisy, so they get EMA smoothing.

    This is the exact variant that uses full SVD on both M and A, corresponding
    to METHOD 1 (compute_exact_p_star) from the RFR benchmark.

    State variables (namespaced under "sv_state"):
    - S_A_ema : Tensor - EMA of full activation singular values
    - C_diag_ema : Tensor - EMA of alignment coefficients

    Complexity: O(mn·min(m,n)) per parameter per update.
    """

    STATE_NAMESPACE = "sv_state"

    def create_state(self, state):
        # Lazily initialized on first call
        pass

    def update_and_compute_p(
        self,
        state,
        grad_2d,
        act_2d,
        use_gram,
        nesterov,
        mom_2d=None,
        beta1=None,
        grad_orig=None,
        act_orig=None,
        mom_orig=None,
        save_path=None,
        **kwargs,
    ):
        if mom_2d is None or beta1 is None:
            raise ValueError("ExactMomentumPApproximator requires mom_2d and beta1")

        # Promote grad/act/mom to a common dtype before any ops between them.
        grad_2d, act_2d, mom_2d = _promote_common(grad_2d, act_2d, mom_2d)

        # Align persisted EMA state if it was created in a different dtype.
        if self.STATE_NAMESPACE in state:
            svs = state[self.STATE_NAMESPACE]
            target = grad_2d.dtype
            for k, v in list(svs.items()):
                if isinstance(v, torch.Tensor) and v.dtype != target:
                    svs[k] = v.to(target)

        # Expected standard momentum (M_t)
        M_t = mom_2d.lerp(grad_2d, 1 - beta1)
        if nesterov:
            mom_2d_expected = M_t.lerp(grad_2d, 1 - beta1)
        else:
            mom_2d_expected = M_t

        # Prepare data for saving (before SVD computations)
        if save_path is not None:
            grad_to_save = grad_orig if grad_orig is not None else grad_2d
            act_to_save = act_orig if act_orig is not None else act_2d
            if mom_orig is not None and grad_orig is not None:
                mom_expected_orig = mom_orig.lerp(grad_orig, 1 - beta1)
            else:
                mom_expected_orig = mom_2d_expected

        # Full SVD on momentum (instantaneous, no EMA)
        U_M, S_M, Vh_M = torch.linalg.svd(mom_2d_expected.float(), full_matrices=False)

        # Alignment coefficients (instantaneous, will be EMA'd)
        C_diag = torch.einsum("mi,mn,ni->i", U_M, grad_2d, Vh_M.T)

        # Full SVD on activations (instantaneous, will be EMA'd)
        if use_gram:
            eigvals = torch.linalg.eigvalsh(act_2d)
            eigvals = torch.clamp(eigvals, min=0.0)
            S_A = torch.sqrt(eigvals).flip(0)
        else:
            S_A = torch.linalg.svdvals(act_2d)

        # Apply EMA to noisy quantities (activations and alignment)
        if self.STATE_NAMESPACE not in state:
            state[self.STATE_NAMESPACE] = {
                "S_A_ema": S_A.clone().detach(),
                "C_diag_ema": C_diag.clone().detach(),
            }
        else:
            svs = state[self.STATE_NAMESPACE]
            if svs["S_A_ema"].shape == S_A.shape:
                svs["S_A_ema"].lerp_(S_A, 1 - self.sv_momentum)
            else:
                svs["S_A_ema"] = S_A.clone().detach()

            if svs["C_diag_ema"].shape == C_diag.shape:
                svs["C_diag_ema"].lerp_(C_diag, 1 - self.sv_momentum)
            else:
                svs["C_diag_ema"] = C_diag.clone().detach()

        S_A_smooth = state[self.STATE_NAMESPACE]["S_A_ema"]
        C_diag_smooth = state[self.STATE_NAMESPACE]["C_diag_ema"]

        def objective(p):
            k_p = 2.0 * (p + 1.0) / (p - 1.0) if p > 1.0 else float("inf")
            norm_A = (
                S_A_smooth[0]
                if k_p == float("inf")
                else torch.sum(S_A_smooth**k_p) ** (1.0 / k_p)
            )
            fp_val = evaluate_fp(p, C_diag_smooth, S_M, norm_A)
            return -fp_val.item()

        res = minimize_scalar(
            objective, bounds=(self.pmin, self.pmax), method="bounded"
        )
        p_star = res.x

        # Save distributions if requested
        if save_path is not None:
            torch.save(
                {
                    "gradient": grad_to_save.cpu(),
                    "momentum": mom_expected_orig.cpu(),
                    "activation": act_to_save.cpu(),
                    "p_star": p_star,
                },
                save_path,
            )

        return p_star
