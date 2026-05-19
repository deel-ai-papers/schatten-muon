"""
Exact Schatten-Muon variants for ablation/diagnostic purposes.

Changes from the previous version:

  * Preconditioning-aware p* criterion. When a second-order moment is in use,
    the D_precond tensor (same shape as the gradient) is forwarded to the
    tightness approximator via update_p_state so p* is chosen consistently
    with the update the optimizer will actually apply.

  * Muon-matched Frobenius normalization. The final update is rescaled so its
    Frobenius norm equals that of the corresponding Muon update U V^T
    (before the max(1, m/n)**0.5 factor). This isolates the "direction"
    choice (p*, preconditioner) from the "magnitude" choice, so the base
    learning rate remains transferable across p_method and moment_type.

  * alpha_star applied as an EMA-damped LR multiplier, with a warmup period
    before it kicks in. This prevents blow-ups when the tightness EMAs are
    under-converged and yield unreliable alpha_star values in the first
    handful of p* updates.

  * The _graft_exp / scale_ref heuristic is removed. Its role is subsumed
    by Muon-matched Frobenius normalization + alpha_star.

  * The hard p_threshold cutoff in the moment exponent is unchanged here
    (it lives in smuon/moments/base.py); I recommend removing it there.
"""

import torch
from smuon.svs.p_registry import create_p_approximator
from smuon.moments import create_moment, is_parametric, MOMENT_REGISTRY


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


def _exact_schatten_update(G, p):
    """U Sigma^{1/p} V^T via full SVD."""
    orig_dtype = G.dtype
    Gf = G.float()
    U, S, Vh = torch.linalg.svd(Gf, full_matrices=False)
    if p >= 1e6:
        S_pow = torch.ones_like(S)
    else:
        S_pow = S.clamp(min=1e-12) ** (1.0 / p)
    out = (U * S_pow.unsqueeze(-2)) @ Vh
    return out.to(orig_dtype)


def _exact_polar_factor(G):
    """U V^T via full SVD — the Muon update direction."""
    orig_dtype = G.dtype
    Gf = G.float()
    U, _, Vh = torch.linalg.svd(Gf, full_matrices=False)
    out = U @ Vh
    return out.to(orig_dtype)


def _muon_reference_norm(m: int, n: int) -> float:
    """
    Frobenius norm of Muon's U V^T update (before the max(1, m/n)**0.5 scale).
    For an (m, n) matrix with r = min(m, n) orthonormal singular vectors,
    ||U V^T||_F = sqrt(r).
    """
    return float(min(m, n)) ** 0.5


class _MuonBaseWithTracking(torch.optim.Optimizer):
    """
    Shared scaffolding for the exact-SVD diagnostic optimizers. Not intended
    to be instantiated directly — subclass and implement _compute_muon_update.
    """

    def __init__(
        self,
        param_groups,
        param_names=None,
        pmin=1.02,
        pmax=50.0,
        eps=1e-5,
        init_p="pmax",
        p_method="exact_tightness",
        subsampling_ratio=0.1,
        moment_type="none",
        use_bias_correction=True,
        use_alpha_star=True,
        match_muon_frobenius=False,
        precond_aware_p=True,
        alpha_star_warmup=100,  # p* updates before alpha_star is applied
        alpha_star_ema_beta=0.9,  # smoothing of per-layer alpha_star over time
        alpha_star_clip=(0.5, 2.0),  # hard clip on the per-step EMA value
    ):
        self._param_names = param_names or {}
        self._pmin = pmin
        self._pmax = pmax
        self._init_p = {"pmax": pmax, "pmin": pmin}[init_p]
        self._p_method = p_method
        self._subsampling_ratio = subsampling_ratio
        self._eps = eps

        if moment_type != "none" and moment_type not in MOMENT_REGISTRY:
            raise ValueError(
                f"Unknown moment_type '{moment_type}'. "
                f"Choose from: {list(MOMENT_REGISTRY.keys()) + ['none']}"
            )
        self._moment_type = moment_type
        self._is_parametric = is_parametric(moment_type)
        self._use_bias_correction = use_bias_correction
        self._use_alpha_star = use_alpha_star
        self._match_muon_frobenius = match_muon_frobenius
        self._precond_aware_p = precond_aware_p

        # Stability knobs for alpha_star
        self._alpha_star_warmup = int(alpha_star_warmup)
        self._alpha_star_ema_beta = float(alpha_star_ema_beta)
        lo, hi = alpha_star_clip
        assert (
            0.0 < lo <= 1.0 <= hi
        ), f"alpha_star_clip must be (lo, hi) with lo <= 1 <= hi, got {alpha_star_clip}"
        self._alpha_star_clip = (float(lo), float(hi))

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("sv_momentum", 0.95)
                group.setdefault("beta2", 0.999)
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("lr_multiplier", 1.0)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)

        super().__init__(param_groups, dict())

    # ------------------------------------------------------------------
    # Moment factory.
    # ------------------------------------------------------------------
    def _get_or_create_moment(self, param, group):
        state = self.state[param]
        if "moment" not in state:
            if self._moment_type == "none":
                state["moment"] = None
            else:
                p_star = state.get("p_star", self._init_p)
                state["moment"] = create_moment(
                    self._moment_type,
                    p=p_star if self._is_parametric else None,
                    beta2=group["beta2"],
                    eps=group["eps"],
                    use_bias_correction=self._use_bias_correction,
                    p_threshold=float("inf"),
                )
        return state["moment"]

    # ------------------------------------------------------------------
    # Build D_precond for the preconditioning-aware p* criterion.
    # Returns None when no moment, or when precond_aware_p is off.
    # The returned tensor has the same shape as grad_2d.
    # ------------------------------------------------------------------
    def _build_d_precond(self, state, moment, grad_2d):
        if moment is None or not self._precond_aware_p:
            return None
        key = getattr(moment, "STATE_KEY", None)
        if key is None or key not in state:
            return None
        v = state[key]
        if v.shape != grad_2d.shape:
            return None
        bias_correction = moment._get_bias_correction(state)
        v_corrected = v * bias_correction
        exponent = moment._get_exponent()
        if exponent == 0.0:
            return None
        return torch.pow(v_corrected + moment.eps, exponent).to(grad_2d.dtype)

    # ------------------------------------------------------------------
    # alpha_star gating + smoothing. Takes the raw alpha_star from the
    # tightness approximator and returns the value to actually multiply
    # into the LR on this step. Handles:
    #   - warmup: return 1.0 until we've seen enough valid alpha_star samples
    #   - EMA: smooth alpha_star over time per layer
    #   - clip: hard bound on per-sample and smoothed values
    #   - None / NaN / non-positive: fall back to current EMA (or 1.0)
    #
    # The counter "alpha_star_n_updates" is incremented only on valid
    # samples so that noisy first-step tightness outputs don't burn the
    # warmup window.
    # ------------------------------------------------------------------
    def _effective_alpha_star(self, state, raw_alpha_star) -> float:
        is_valid = (
            raw_alpha_star is not None
            and isinstance(raw_alpha_star, (int, float))
            and raw_alpha_star == raw_alpha_star  # NaN check
            and raw_alpha_star > 0.0
        )

        if not is_valid:
            # Don't update EMA with garbage; return current effective value.
            return float(state.get("alpha_star_ema", 1.0))

        lo, hi = self._alpha_star_clip
        clipped = max(lo, min(float(raw_alpha_star), hi))

        prev = state.get("alpha_star_ema", 1.0)
        beta = self._alpha_star_ema_beta
        ema = beta * prev + (1.0 - beta) * clipped
        ema = max(lo, min(ema, hi)) / hi  # Normalize so max alpha is 1.0

        state["alpha_star_ema"] = ema
        state["alpha_star_n_updates"] = state.get("alpha_star_n_updates", 0) + 1

        # Warmup: during the first N valid updates, trust alpha_star = 1.
        if state["alpha_star_n_updates"] < self._alpha_star_warmup:
            return 1.0

        return float(ema)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def get_p_state_for_logging(self):
        out = {}
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                state = self.state[p]
                if "p_star" in state:
                    name = self._param_names.get(p, f"param_{list(p.shape)}")
                    out[name] = {
                        "p_star": state["p_star"],
                        "alpha_star": state.get("alpha_star", None),
                        "alpha_star_ema": state.get("alpha_star_ema", None),
                    }
        return out

    # ------------------------------------------------------------------
    # p* update. Threads D_precond through to the tightness approximator.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_p_state(self, activations=None, global_step=None, use_gram=False):
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            sv_beta = group["sv_momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad

                if grad.ndim == 4:
                    grad_2d = grad.reshape(grad.size(0), -1)
                elif grad.ndim == 2:
                    grad_2d = grad
                else:
                    continue

                if activations is None or p not in activations:
                    if "p_star" not in state:
                        state["p_star"] = self._init_p
                    continue

                act = activations.pop(p)

                if use_gram:
                    act_2d = act.float()
                else:
                    if "tightness" in self._p_method:
                        act_2d = act
                    else:
                        act_2d = act.reshape(act.size(0), -1) if act.ndim > 2 else act

                if torch.isnan(grad_2d).any() or torch.isnan(act_2d).any():
                    continue

                if "p_approximator" not in state:
                    state["p_approximator"] = create_p_approximator(
                        method=self._p_method,
                        pmin=self._pmin,
                        pmax=self._pmax,
                        subsampling_ratio=self._subsampling_ratio,
                        sv_momentum=sv_beta,
                    )
                    state["p_approximator"].create_state(state)

                approximator = state["p_approximator"]

                kwargs = {}
                if any(m in self._p_method for m in ["momentum", "tightness"]):
                    mom_buf = state.get("momentum_buffer", torch.zeros_like(grad))
                    muon_step = state.get("muon_step", 0) + 1
                    bc = 1.0 - group["momentum"] ** muon_step
                    mom_buf = mom_buf / bc
                    mom_2d = mom_buf.reshape(mom_buf.size(0), -1)
                    kwargs["mom_2d"] = mom_2d
                    kwargs["beta1"] = group["momentum"]

                if "tightness" in self._p_method:
                    moment = self._get_or_create_moment(p, group)
                    if moment is not None and self._is_parametric:
                        moment.p = state.get("p_star", self._init_p)
                    D_precond = self._build_d_precond(state, moment, grad_2d)
                    if D_precond is not None:
                        kwargs["D_precond"] = D_precond

                if "tightness" in self._p_method:
                    result = approximator.update_and_compute_p(
                        state,
                        grad_2d,
                        act_2d,
                        use_gram=use_gram,
                        nesterov=True,
                        **kwargs,
                    )
                    if result is None:
                        continue
                    p_star, alpha_star = result
                else:
                    p_star = approximator.update_and_compute_p(
                        state,
                        grad_2d,
                        act_2d,
                        use_gram=use_gram,
                        nesterov=True,
                        **kwargs,
                    )
                    alpha_star = None

                if p_star is None:
                    continue

                state["p_star"] = p_star
                # Store raw alpha_star for logging. Smoothed / gated version
                # is stored under alpha_star_ema by _effective_alpha_star.
                state["alpha_star"] = alpha_star

        if activations is not None:
            activations.clear()

    # ------------------------------------------------------------------
    # Step driver.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adam(group)

        return loss

    def _step_muon(self, group):
        beta1 = group["momentum"]

        for param in group["params"]:
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[param]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param)
                state["muon_step"] = 0
                state["p_star"] = self._init_p

            state["muon_step"] += 1
            step = state["muon_step"]
            p_star = state.get("p_star", self._init_p)
            raw_alpha_star = state.get("alpha_star", None)

            mom_buf = state["momentum_buffer"]
            mom_buf.lerp_(grad, 1 - beta1)
            bc = 1.0 - beta1**step
            mom_buf_corrected = mom_buf / bc
            nesterov = mom_buf_corrected.lerp(grad, 1 - beta1)

            orig_shape = nesterov.shape
            if nesterov.ndim > 2:
                nesterov = nesterov.reshape(nesterov.size(0), -1)

            # Moment: update variance on the RAW gradient (so v matches Adam's
            # convention and the scale-invariance argument holds), but apply
            # the preconditioner to the Nesterov-corrected momentum that goes
            # into the SVD.
            moment = self._get_or_create_moment(param, group)
            if moment is not None and self._is_parametric:
                moment.p = p_star

            if moment is not None:
                grad_2d_for_moment = (
                    grad.reshape(grad.size(0), -1) if grad.ndim > 2 else grad
                )
                D_t = moment.update(state, grad_2d_for_moment)
                g_tilde = D_t * nesterov
            else:
                D_t = None
                g_tilde = nesterov

            if g_tilde.norm() < 1e-8:
                continue

            # Spectral update on the preconditioned momentum.
            update = self._compute_muon_update(g_tilde, p_star)

            # Symmetric preconditioning: map-out.
            if D_t is not None:
                update = D_t * update

            # Muon-matched Frobenius normalization. Isolates direction from
            # magnitude so the base LR stays transferable.
            if self._match_muon_frobenius:
                m, n = update.size(-2), update.size(-1)
                target_norm = _muon_reference_norm(m, n)
                cur_norm = update.norm()
                update = update * (target_norm / (cur_norm + 1e-8))

            # Muon's tall/fat rescaling (same as the base Muon code).
            update = update * (max(1, update.size(-2) / update.size(-1)) ** 0.5)

            # alpha_star as a theory-derived per-layer LR multiplier, passed
            # through warmup + EMA + clipping to prevent blow-ups.
            if self._use_alpha_star:
                alpha_eff = self._effective_alpha_star(state, raw_alpha_star)
                effective_lr = group["lr"] * alpha_eff
            else:
                effective_lr = group["lr"]

            if group["weight_decay"] > 0:
                param.mul_(1 - effective_lr * group["weight_decay"])

            param.add_(update.reshape(orig_shape), alpha=-effective_lr)

    def _step_adam(self, group):
        lr_mult = group.get("lr_multiplier", 1.0)
        lr = group["lr"] * lr_mult
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] = 0
            state["step"] += 1
            update = adam_update(
                param.grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
                group["betas"],
                group["eps"],
            )
            param.mul_(1 - lr * group["weight_decay"])
            param.add_(update, alpha=-lr)

    def _compute_muon_update(self, nesterov_2d, p_star):
        raise NotImplementedError


class ExactSMuonWithAuxAdam(_MuonBaseWithTracking):
    """
    Exact-SVD Schatten-p Muon (Σ^{1/p*} via torch.linalg.svd). Inherits
    preconditioning-aware p*, alpha_star with warmup/EMA/clipping, and
    Muon-matched Frobenius normalization from the base class.
    """

    def _compute_muon_update(self, nesterov_2d, p_star):
        return _exact_schatten_update(nesterov_2d, p_star)


class ExactMuonWithOptimalPLogging(_MuonBaseWithTracking):
    """
    Applies the exact Muon update (polar factor U V^T) at every step but
    still computes and logs p*, alpha*. The applied update ignores p*.

    Useful for the retroaction test: does running pure Muon drive p*(t) up
    over training?
    """

    def _compute_muon_update(self, nesterov_2d, p_star):
        return _exact_polar_factor(nesterov_2d)
