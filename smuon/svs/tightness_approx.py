"""
Tightness-corrected p* via top-k SVD with power-law tail extrapolation.

Root-cause fix (identified via debug prints):
  The previous version EMA'd total_GM and MA_Fsq separately, then computed:
      tail_GM = totGM_ema - top_sum_C_fresh
  This mixes a lagged EMA with a fresh quantity.  When total_GM is rising
  (early training), totGM_ema < total_GM_fresh ≈ top_sum_C, producing a
  spuriously large negative tail_GM.  At high p, tail_s_{1/p}/tail_s_1 >> 1
  (small tail sigmas get hugely amplified), so N_tail(p→∞) = tail_GM × giant
  blows up negatively while N_top grows positively — catastrophic cancellation
  that makes J(p) meaningless.

  Fix: compute tail_GM = total_GM_fresh - top_sum_C_fresh (both from the
  same step, same SVD), then EMA tail_GM itself.  This guarantees the anchor
  is always self-consistent.  Likewise for tail_D.

Architecture:
  * Power-law tail spectrum: fits sigma_i ≈ A i^{-alpha} to the trailing
    singular values and uses the fit to compute sum_{tail} sigma^q for any q.
  * Constant-alignment-density tail model:
        N_tail(p) = tail_GM * tail_sigma^{1/p} / tail_sigma^1
        D_tail(p) = tail_D  * tail_sigma^{2/p} / tail_sigma^2
  * C_diag used fresh each step (avoids EMA sign-flip decay).
  * B_diag EMA'd (non-negative, safe to smooth).
  * tail_GM and tail_D EMA'd after being computed from consistent fresh values.

State:
    B   : EMA of top-k activation norms  (non-negative)
    tGM : EMA of tail_GM = <G,M> - sum_top C_i sigma_i  (can be negative)
    tD  : EMA of tail_D  = ||MA||^2 - sum_top B_i sigma_i^2  (clamped ≥ 0)
"""

import torch
from scipy.optimize import minimize_scalar

from smuon.svs.base import PApproximator
from smuon.svs.tightness_exact import _promote_common, _orient_activation


# ---------------------------------------------------------------------------
# Power-law tail helpers
# ---------------------------------------------------------------------------


def _fit_power_law(S: torch.Tensor, fit_n: int = 15):
    """
    OLS fit of  log(sigma_i) = log(A) - alpha * log(i)  to the last `fit_n`
    elements of S (1-indexed ranks within S).
    Returns (A, alpha) as float64 tensors; alpha clamped to [0.05, 20].
    """
    k = len(S)
    fit_n = max(3, min(fit_n, k))
    device, f64 = S.device, torch.float64

    ranks = torch.arange(k - fit_n + 1, k + 1, dtype=f64, device=device)
    log_r = torch.log(ranks)
    log_s = torch.log(S[-fit_n:].to(f64).clamp(min=1e-30))

    lr_c = log_r - log_r.mean()
    ls_c = log_s - log_s.mean()
    alpha = -(lr_c @ ls_c) / (lr_c @ lr_c + 1e-30)
    alpha = alpha.clamp(min=0.05, max=20.0)
    log_A = log_s.mean() + alpha * log_r.mean()

    return torch.exp(log_A), alpha


def _tail_sigma_q(
    A: torch.Tensor, alpha: torch.Tensor, k: int, min_dim: int, q: float
) -> torch.Tensor:
    """
    Estimate  sum_{i=k+1}^{min_dim} (A * i^{-alpha})^q  by direct summation.
    At q ≈ 0 each sigma^q → 1, so the sum → (min_dim - k).
    """
    d_tail = min_dim - k
    if d_tail <= 0:
        return A.new_zeros(())
    if abs(q) < 1e-7:
        return A.new_tensor(float(d_tail))
    ranks = torch.arange(k + 1, min_dim + 1, dtype=A.dtype, device=A.device)
    return A.pow(q) * ranks.pow(-alpha * q).sum()


# ---------------------------------------------------------------------------
# Approximator
# ---------------------------------------------------------------------------


class ApproxTightnessPApproximator(PApproximator):
    """
    Approximates p* using a truncated top-k SVD plus a power-law tail model.
    """

    STATE_NAMESPACE = "sv_state"
    _SVD_OVERSAMPLE = 20
    _SVD_NITER = 4
    _PL_FIT_N = 15

    def create_state(self, state):
        if self.STATE_NAMESPACE not in state:
            state[self.STATE_NAMESPACE] = {}

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
            raise ValueError("ApproxTightnessPApproximator requires mom_2d and beta1")

        # ── Step math in FP32 ────────────────────────────────────────────────
        grad_2d, act_2d, mom_2d = _promote_common(
            grad_2d, act_2d, mom_2d, min_dtype=torch.float32
        )
        M_t = mom_2d.lerp(grad_2d, 1 - beta1)
        M_expected = M_t.lerp(grad_2d, 1 - beta1) if nesterov else M_t

        M_for_svd = M_expected
        G_for_C = grad_2d

        m_dim, n_dim = M_for_svd.shape
        min_dim = min(m_dim, n_dim)
        k = min(
            min_dim,
            max(kwargs.get("min_rank", 50), int(min_dim * self.subsampling_ratio)),
        )

        # ── Truncated SVD (FP32) ─────────────────────────────────────────────
        q_svd = min(k + self._SVD_OVERSAMPLE, min_dim)
        U_raw, S_raw, V_raw = torch.svd_lowrank(
            M_for_svd, q=q_svd, niter=self._SVD_NITER
        )
        U_M = U_raw[:, :k]
        V_M = V_raw[:, :k]
        S_M = S_raw[:k].to(torch.float64)

        # ── Top-k projections ────────────────────────────────────────────────
        A_oriented = _orient_activation(V_M, act_2d, use_gram)

        C_diag = torch.einsum("mi,mn,ni->i", U_M, G_for_C, V_M).to(torch.float64)

        if use_gram:
            VtGA = V_M.t() @ A_oriented
            B_diag = (VtGA * V_M.t()).sum(-1).to(torch.float64).clamp(min=0.0)
        else:
            VtA = V_M.t() @ A_oriented
            B_diag = (VtA**2).sum(-1).to(torch.float64).clamp(min=0.0)

        # ── Consistent fresh tail anchors ────────────────────────────────────
        # KEY FIX: compute tail_GM and tail_D using quantities from the SAME
        # step (both fresh).  Mixing EMA-totGM with fresh top_sum_C caused a
        # lag-induced sign error that corrupted the tail model at high p.
        total_GM = (G_for_C.double() * M_for_svd.double()).sum()
        MA_Fsq = (M_for_svd.double() @ A_oriented.double()).pow(2).sum()

        top_sum_C = (C_diag * S_M).sum()  # fresh, p=1
        top_sum_B = (B_diag * S_M.pow(2)).sum()  # fresh, p=1

        tail_GM_fresh = total_GM - top_sum_C  # self-consistent; can be negative
        tail_D_fresh = (MA_Fsq - top_sum_B).clamp(min=0.0)

        # ── Power-law fit ─────────────────────────────────────────────────────
        A_pl, alpha_pl = _fit_power_law(S_M, fit_n=self._PL_FIT_N)

        if k < min_dim:
            tail_s1 = _tail_sigma_q(A_pl, alpha_pl, k, min_dim, 1.0)
            tail_s2 = _tail_sigma_q(A_pl, alpha_pl, k, min_dim, 2.0)
        else:
            tail_s1 = tail_s2 = None

        # ── EMA updates ──────────────────────────────────────────────────────
        svs = state[self.STATE_NAMESPACE]
        mv = 1.0 - self.sv_momentum

        if "B" not in svs or svs["B"].shape != B_diag.shape:
            svs.update(
                B=B_diag.clone(),
                tGM=tail_GM_fresh.clone(),
                tD=tail_D_fresh.clone(),
            )
        else:
            svs["B"].lerp_(B_diag, mv)
            svs["tGM"].lerp_(tail_GM_fresh, mv)
            svs["tD"].lerp_(tail_D_fresh, mv)

        B_s = svs["B"]
        tail_GM = svs["tGM"]
        tail_D = svs["tD"]

        # ── J(p) = N(p)^2 / D(p) ────────────────────────────────────────────
        def _compute_ND(p: float):
            inv_p = 1.0 / p
            # N uses fresh C_diag; D uses EMA'd B_s
            N = (C_diag * S_M.pow(inv_p)).sum()
            D = (B_s * S_M.pow(2 * inv_p)).sum()

            if k < min_dim:
                ts_invp = _tail_sigma_q(A_pl, alpha_pl, k, min_dim, inv_p)
                ts_2invp = _tail_sigma_q(A_pl, alpha_pl, k, min_dim, 2 * inv_p)
                amp_N = ts_invp / (tail_s1 + 1e-30)
                amp_D = ts_2invp / (tail_s2 + 1e-30)
                # Cap amplification: beyond ~500× the anchor is too uncertain to trust
                amp_N = amp_N.clamp(max=500.0)
                amp_D = amp_D.clamp(max=500.0)
                N = N + tail_GM * amp_N
                D = D + tail_D * amp_D

            return N, D

        n1, d1 = _compute_ND(1.0)
        scale_inv = 1.0 / ((n1.pow(2) / (d1 + 1e-12)).abs().item() + 1e-30)

        def objective(p: float) -> float:
            n, d = _compute_ND(p)
            return -((n.pow(2) / (d + 1e-12)).item() * scale_inv)

        res = minimize_scalar(
            objective, bounds=(self.pmin, self.pmax), method="bounded"
        )
        p_star = float(res.x)

        # ── alpha_star ────────────────────────────────────────────────────────
        n_p, d_p = _compute_ND(p_star)
        n_inf, d_inf = _compute_ND(1e6)
        eta_p = (n_p / (2 * d_p + 1e-12)).item()
        eta_inf = (n_inf / (2 * d_inf + 1e-12)).item()
        alpha_star = (
            1.0
            if (eta_inf <= 0 or eta_p <= 0)
            else float(max(0.5, min(eta_p / eta_inf, 2.0)))
        )

        return p_star, alpha_star
