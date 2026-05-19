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


class MomentumApproxPApproximator(PApproximator):
    """
    Momentum-aligned p* approximation with selective EMA smoothing.

    Key insight: Momentum M_t is naturally smooth (high beta1 ≈ 0.95), so we use
    its instantaneous SVs without EMA. Activations and alignment (grad·mom) are
    noisy, so they get EMA smoothing.

    State variables (namespaced under "sv_state"):
    - S_A : Tensor - EMA of top-k activation singular values
    - tail_A_2 : Tensor (scalar) - EMA of activation tail energy
    - C_diag : Tensor - EMA of alignment coefficients (grad projected onto momentum basis)
    - tail_GM : Tensor (scalar) - EMA of tail alignment energy

    Complexity: O(mn·k) where k = subsampling_ratio * min(m,n).
    """

    STATE_NAMESPACE = "sv_state"

    def create_state(self, state):
        # Lazily initialized on first call
        pass

    def update_and_compute_p(
        self, state, grad_2d, act_2d, use_gram, mom_2d=None, beta1=None, **kwargs
    ):
        if mom_2d is None or beta1 is None:
            raise ValueError("MomentumApproxPApproximator requires mom_2d and beta1")

        grad_2d, act_2d, mom_2d = _promote_common(
            grad_2d, act_2d, mom_2d, min_dtype=torch.float32
        )

        if self.STATE_NAMESPACE in state:
            svs = state[self.STATE_NAMESPACE]
            target = grad_2d.dtype
            for k, v in list(svs.items()):
                if isinstance(v, torch.Tensor) and v.dtype != target:
                    svs[k] = v.to(target)

        # Expected M_t (out-of-place to avoid corrupting step logic)
        mom_2d_expected = mom_2d.lerp(grad_2d, 1 - beta1)

        M_xdim, M_ydim = mom_2d_expected.shape
        A_xdim, A_ydim = act_2d.shape

        k_top_M = max(1, int(min(M_xdim, M_ydim) * self.subsampling_ratio))
        k_top_A = max(1, int(min(A_xdim, A_ydim) * self.subsampling_ratio))

        # 2. SVD on Momentum
        U_M, S_M, V_M = torch.svd_lowrank(mom_2d_expected, q=k_top_M)

        # 3. SVD on Activations (safely handle distributed Gram matrices)
        if use_gram:
            _, L_A, _ = torch.svd_lowrank(act_2d, q=k_top_A)
            S_A = torch.sqrt(torch.relu(L_A))
            frob_A_sq = torch.trace(act_2d)
            d_A = act_2d.shape[0] - k_top_A
        else:
            _, S_A, _ = torch.svd_lowrank(act_2d, q=k_top_A)
            frob_A_sq = torch.sum(act_2d**2)
            d_A = min(A_xdim, A_ydim) - k_top_A

        # 4. Pure M values (naturally smooth - no EMA needed)
        frob_M_sq = torch.sum(mom_2d_expected**2)
        tail_M_2 = torch.clamp(frob_M_sq - torch.sum(S_M**2), min=0.0)
        d_M = min(M_xdim, M_ydim) - k_top_M

        # 5. Instantaneous A & alignment values (highly noisy - needs EMA)
        tail_A_2 = torch.clamp(frob_A_sq - torch.sum(S_A**2), min=0.0)

        C_diag = torch.einsum("mi,mn,ni->i", U_M, grad_2d, V_M)
        total_GM = torch.sum(grad_2d * mom_2d_expected)
        top_GM = torch.sum(C_diag * S_M)
        tail_GM = torch.clamp(total_GM - top_GM, min=0.0)

        # 6. Update EMA trackers
        if self.STATE_NAMESPACE not in state:
            state[self.STATE_NAMESPACE] = {
                "S_A": S_A.clone().detach(),
                "tail_A_2": tail_A_2.clone().detach(),
                "C_diag": C_diag.clone().detach(),
                "tail_GM": tail_GM.clone().detach(),
            }
        else:
            svs = state[self.STATE_NAMESPACE]
            if svs["S_A"].shape == S_A.shape:
                svs["S_A"].lerp_(S_A, 1 - self.sv_momentum)
                svs["C_diag"].lerp_(C_diag, 1 - self.sv_momentum)
            else:
                svs["S_A"] = S_A.clone().detach()
                svs["C_diag"] = C_diag.clone().detach()

            svs["tail_A_2"].lerp_(tail_A_2, 1 - self.sv_momentum)
            svs["tail_GM"].lerp_(tail_GM, 1 - self.sv_momentum)

        # 7. Extract smoothed values
        S_A_smooth = state[self.STATE_NAMESPACE]["S_A"]
        tail_A_2_smooth = state[self.STATE_NAMESPACE]["tail_A_2"]
        C_diag_smooth = state[self.STATE_NAMESPACE]["C_diag"]
        tail_GM_smooth = state[self.STATE_NAMESPACE]["tail_GM"]

        avg_sigma_A = torch.sqrt(tail_A_2_smooth / max(d_A, 1)) + 1e-12
        avg_sigma_M = torch.sqrt(tail_M_2 / max(d_M, 1)) + 1e-12

        # 8. Objective function
        def objective(p):
            k_p = 2.0 * (p + 1.0) / (p - 1.0) if p > 1.0 else float("inf")
            power_M = (p + 1.0) / p
            p_safe = min(p, 10000.0)

            if k_p == float("inf"):
                norm_A = S_A_smooth[0]
            elif k_p == 2.0:
                norm_A = torch.sqrt(torch.sum(S_A_smooth**2) + tail_A_2_smooth)
            else:
                tail_A_k = d_A * (avg_sigma_A**k_p) if d_A > 0 else 0.0
                norm_A = (torch.sum(S_A_smooth**k_p) + tail_A_k) ** (1.0 / k_p)

            if power_M == 2.0:
                sum_M = frob_M_sq
            else:
                tail_M_k = d_M * (avg_sigma_M**power_M) if d_M > 0 else 0.0
                sum_M = torch.sum(S_M**power_M) + tail_M_k

            term2 = sum_M ** (2.0 / (p + 1.0))

            tail_num = (
                tail_GM_smooth * (avg_sigma_M ** (1.0 / p_safe - 1.0))
                if d_M > 0
                else 0.0
            )
            num_val = torch.sum(C_diag_smooth * (S_M ** (1.0 / p_safe))) + tail_num
            num = num_val**2

            den = (norm_A**2) * term2
            return -(num / (den + 1e-12)).item()

        # 9. Solve for p*
        res = minimize_scalar(
            objective, bounds=(self.pmin, self.pmax), method="bounded"
        )
        return res.x
