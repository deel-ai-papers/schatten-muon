"""
Schatten-Muon optimizer for distributed training.

Simplified version using fractional_power_from_polar instead of Newton-Schulz with SLSQP coefficients.
Includes batched operations for identically shaped parameter tensors and supports multi-GPU via Gram trick.

Update magnitudes are pinned via the kappa^(1/p) scheme so that:
  - at p -> infty, the per-step magnitude matches Muon's natural step (eta_muon * rho_muon)
  - at p  = 1   , the per-step magnitude matches Adam's natural step (eta_adam * rho_adam)
  - intermediate p smoothly interpolates between the two.

The constant kappa is derived from the auxiliary Adam group's learning rate (the
optimizer already has Adam configured for scalar parameters), so no extra
hyperparameter is introduced.
"""

import math
import torch
import torch.distributed as dist

from smuon.svs.p_registry import create_p_approximator
from smuon.p_root import fractional_power_from_polar

from smuon.moments import create_moment, is_parametric, MOMENT_REGISTRY


def schatten_update_batched(
    G,
    p_values,
    epsilon=1e-3,
    max_extra_iters=5,
    linf_tol=5e-3,
    lambda_reg=1e-4,
    pmax=10.0,
    normalize=False,
):
    is_single = G.ndim == 2
    original_dtype = G.dtype

    if is_single:
        G = G.unsqueeze(0)
        if not isinstance(p_values, torch.Tensor):
            p_values = torch.tensor([p_values], device=G.device, dtype=torch.float32)
        elif p_values.ndim == 0:
            p_values = p_values.unsqueeze(0)
    else:
        if not isinstance(p_values, torch.Tensor):
            p_values = torch.full(
                (G.size(0),), p_values, device=G.device, dtype=torch.float32
            )
        elif p_values.ndim == 0:
            p_values = p_values.expand(G.size(0))

    p_values = p_values.to(G.device)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        G = G.transpose(-2, -1)

    p_orders = 1.0 / p_values

    output, polar_scale = fractional_power_from_polar(
        G,
        p_orders,
        eigen_floor=epsilon,
        max_extra_iters=max_extra_iters,
        linf_tol=linf_tol,
        lambda_reg=lambda_reg,
        p_threshold=1.01 * (1 / pmax),
        normalize=normalize,
    )

    if transposed:
        output = output.transpose(-2, -1)

    if is_single:
        output = output.squeeze(0)
        polar_scale = polar_scale.squeeze(0)

    return output.to(original_dtype), polar_scale


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Standard Adam update."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


def _muon_reference_norm(m: int, n: int) -> float:
    """
    Frobenius norm of Muon's U V^T update (before the max(1, m/n)**0.5 scale).
    For an (m, n) matrix with r = min(m, n) orthonormal singular vectors,
    ||U V^T||_F = sqrt(r).
    """
    return float(min(m, n)) ** 0.5


def _rho_adam(beta1: float, regime: str = "noisy") -> float:
    """
    Per-coordinate RMS of Adam's update m_hat / sqrt(v_hat), in the steady state
    where gradients are i.i.d. with mean mu and variance sigma^2.

    regime="noisy"  : sigma >> mu, RMS = sqrt((1 - beta1) / (1 + beta1)).
    regime="determ" : sigma << mu, RMS = 1.

    The "noisy" choice is the conservative default and matches the regime in
    which Adam-like behavior is actually desirable (otherwise SGD or Muon already
    suffice). The constant is layer- and parameter-independent.
    """
    if regime == "determ":
        return 1.0
    return math.sqrt((1.0 - beta1) / (1.0 + beta1))


class _KappaPinMixin:
    """
    Shared logic for kappa-pinning the post-Schatten update so that:
      ||Delta W||_F = sqrt(min(m,n)) * kappa^(1/p)

    where
      kappa = (eta_adam * rho_adam) / (eta_muon * rho_muon)
            = (eta_adam * rho_adam / eta_muon) * sqrt(max(m,n)).

    At p = infty: ||Delta W||_F = sqrt(min(m,n))   (Muon's raw UV^T norm)
    At p = 1    : ||Delta W||_F = sqrt(min(m,n)) * kappa
                                = (eta_adam * rho_adam / eta_muon) * sqrt(m*n)
                  -> per-step magnitude eta_muon * RMS = eta_adam * rho_adam (Adam's step).

    Multiplying by Muon's tall/fat rescaling sqrt(max(1, m/n)) is intentionally
    NOT applied here: rho_muon = 1/sqrt(max(m,n)) already encodes the size scaling
    and re-applying it would double-correct.
    """

    # Subclasses must set these in __init__:
    #   self._kappa_adam_regime: "noisy" or "determ"
    #   self._kappa_enabled    : bool
    #   self._pmax             : float

    def _aux_adam_step_magnitude(self):
        """
        Returns eta_adam * rho_adam from the first non-Muon param group, or None
        if no Adam group exists (in which case kappa-pinning falls back to pure
        Muon RMS, equivalent to ||Delta W||_F = sqrt(min(m,n)) for all p).
        """
        for g in self.param_groups:
            if not g.get("use_muon", False):
                lr_adam = g["lr"] * g.get("lr_multiplier", 1.0)
                beta1 = g["betas"][0]
                rho = _rho_adam(beta1, regime=self._kappa_adam_regime)
                return float(lr_adam * rho)
        return None

    def _target_frobenius(self, group, m, n, p_star):
        """
        Compute target ||Delta W||_F for kappa-pinning.

        Returns sqrt(min(m,n)) * kappa^(1/p_star) when kappa-pinning is on and an
        Adam group is configured, else sqrt(min(m,n)) (Muon's natural Frobenius
        norm). Independent of the gradient spectrum, so the optimal lr transfers
        across p.
        """
        ref = _muon_reference_norm(m, n)  # sqrt(min(m, n))
        if not self._kappa_enabled:
            return ref

        # Always pin to a p-independent target so one eta transfers across p.
        # The target itself depends on whether we have an Adam endpoint to
        # interpolate toward.
        if self._moment_type == "none":
            # No Adam endpoint -> no interpolation, just pin to Muon's RMS.
            return ref

        # Parametric moment: p=1 recovers Adam direction. Interpolate magnitude
        # between Muon (p->inf) and Adam (p=1) endpoints via kappa^(1/p).
        adam_step = self._aux_adam_step_magnitude()
        if adam_step is None:
            return ref
        eta_muon = group["lr"]
        if eta_muon <= 0.0:
            return ref
        rho_muon = 1.0 / math.sqrt(max(m, n))
        kappa = max(adam_step / (eta_muon * rho_muon), 1e-12)
        p_safe = max(float(p_star), 1.0)
        return ref * (kappa ** (1.0 / p_safe))


class SMuonWithAuxAdam(_KappaPinMixin, torch.optim.Optimizer):
    """
    Schatten-Muon optimizer with adaptive p selection for multi-GPU training.

    Uses fractional_power_from_polar for computing Σ^{1/p} via polar decomposition,
    replacing Newton-Schulz iterations with SLSQP coefficients. Supports distributed
    training via Gram matrix trick for tracking activation singular values across devices.

    Update magnitudes are pinned with kappa^(1/p) (see _KappaPinMixin) so that
    a single eta_muon transfers across the whole p in [1, pmax] family while
    recovering Muon's step at p -> infty and Adam's step at p = 1.
    """

    def __init__(
        self,
        param_groups,
        param_names=None,
        pmin=1.02,
        pmax=10.0,
        eps=1e-5,
        init_p="pmax",
        p_method="exact_momentum",
        subsampling_ratio=0.1,
        moment_type="none",
        use_bias_correction=True,
        kappa_pin=True,
        kappa_adam_regime="noisy",
    ):
        self._param_names = param_names or {}
        self._pmin = pmin
        self._pmax = pmax
        self._init_p = {
            "pmax": pmax,
            "pmin": pmin,
        }[init_p]
        self._eps = eps
        self._p_method = p_method
        self._subsampling_ratio = subsampling_ratio
        self._normalize = False if moment_type != "none" else True

        if moment_type != "none" and moment_type not in MOMENT_REGISTRY:
            raise ValueError(
                f"Unknown moment_type '{moment_type}'. "
                f"Choose from: {list(MOMENT_REGISTRY.keys()) + ['none']}"
            )

        self._moment_type = moment_type
        self._is_parametric = is_parametric(moment_type)
        self._use_bias_correction = use_bias_correction

        self._kappa_enabled = bool(kappa_pin)
        if kappa_adam_regime not in ("noisy", "determ"):
            raise ValueError(
                f"kappa_adam_regime must be 'noisy' or 'determ', got {kappa_adam_regime!r}"
            )
        self._kappa_adam_regime = kappa_adam_regime

        self._padding_cache = {}

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

    def _get_or_create_moment(self, param, group):
        state = self.state[param]
        if "moment" not in state:
            if self._moment_type == "none":
                state["moment"] = None
            else:
                p_star = state.get("p_star", self._pmin)
                state["moment"] = create_moment(
                    self._moment_type,
                    p=p_star if self._is_parametric else None,
                    beta2=group["beta2"],
                    eps=group["eps"],
                    use_bias_correction=self._use_bias_correction,
                    p_threshold=0.99 * self._pmax,
                )
        return state["moment"]

    def get_p_state_for_logging(self):
        """Return {param_name: {"p_star": float}} for all Muon params."""
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
                    }
        return out

    @torch.no_grad()
    def update_p_state(self, activations=None, use_gram=False):
        """
        Update optimal p* for each Muon parameter using gradient and activation SVs.
        """
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

                # Handle Gram matrices (distributed training)
                if use_gram:
                    gram = act.float()
                    if torch.isnan(gram).any():
                        continue
                    if dist.is_initialized():
                        dist.all_reduce(gram, op=dist.ReduceOp.SUM)
                    act_2d = gram
                else:
                    if "tightness" in self._p_method:
                        act_2d = act
                    else:
                        act_2d = act.reshape(act.size(0), -1) if act.ndim > 2 else act

                if torch.isnan(grad_2d).any() or torch.isnan(act_2d).any():
                    continue

                # Get or create P-approximator
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

                # Prepare method-specific kwargs
                kwargs = {}
                if any([m in self._p_method for m in ["momentum", "tightness"]]):
                    mom_buf = state.get("momentum_buffer", torch.zeros_like(grad))
                    muon_step = state.get("muon_step", 0) + 1
                    bc = 1.0 - group["momentum"] ** muon_step
                    mom_buf = mom_buf / bc
                    mom_2d = mom_buf.reshape(mom_buf.size(0), -1)
                    kwargs["mom_2d"] = mom_2d
                    kwargs["beta1"] = group["momentum"]

                # Compute p*
                if "tightness" in self._p_method:
                    p_star, _ = approximator.update_and_compute_p(
                        state,
                        grad_2d,
                        act_2d,
                        use_gram=use_gram,
                        nesterov=True,
                        **kwargs,
                    )
                else:
                    p_star = approximator.update_and_compute_p(
                        state,
                        grad_2d,
                        act_2d,
                        use_gram=use_gram,
                        nesterov=True,
                        **kwargs,
                    )

                if p_star is None:
                    continue

                state["p_star"] = p_star

        if activations is not None:
            activations.clear()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        muon_shape_groups = {}

        # Pass 1: Process all parameters
        for group in self.param_groups:
            params = group["params"]
            num_padding = world_size - len(params) % world_size
            if num_padding == world_size:
                num_padding = 0

            cache_key = id(group)
            if (
                cache_key not in self._padding_cache
                or len(self._padding_cache[cache_key]) != num_padding
            ):
                self._padding_cache[cache_key] = [
                    torch.zeros_like(params[-1]) for _ in range(num_padding)
                ]

            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    if not group["use_muon"]:
                        # Adam update
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1

                        update = adam_update(
                            p.grad,
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            state["step"],
                            group["betas"],
                            group["eps"],
                        )
                        lr = group["lr"] * group.get("lr_multiplier", 1.0)
                        p.mul_(1 - lr * group["weight_decay"])
                        p.add_(update, alpha=-lr)
                    else:
                        # Muon update preparation
                        beta1 = group["momentum"]
                        grad = p.grad
                        state = self.state[p]

                        # Initialize state
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["muon_step"] = 0
                            state["p_star"] = self._init_p

                        state["muon_step"] += 1
                        step = state["muon_step"]

                        # Get p_star
                        p_star = state.get("p_star", self._pmin)

                        # Nesterov momentum with bias correction
                        mom_buf = state["momentum_buffer"]
                        mom_buf.lerp_(grad, 1 - beta1)

                        # Bias correction
                        bc = 1.0 - beta1**step
                        mom_buf_corrected = mom_buf / bc

                        # Nesterov lookahead
                        nesterov = mom_buf_corrected.lerp(grad, 1 - beta1)

                        # Reshape to 2D
                        orig_shape = nesterov.shape
                        if nesterov.ndim > 2:
                            nesterov = nesterov.reshape(nesterov.size(0), -1)

                        # Moment: update variance on the RAW gradient (so v matches Adam's
                        # convention and the scale-invariance argument holds), but apply
                        # the preconditioner to the Nesterov-corrected momentum that goes
                        # into the SVD.
                        moment = self._get_or_create_moment(p, group)
                        if moment is not None and self._is_parametric:
                            moment.p = p_star

                        if moment is not None:
                            grad_2d_for_moment = (
                                grad.reshape(grad.size(0), -1)
                                if grad.ndim > 2
                                else grad
                            )
                            D_t = moment.update(state, grad_2d_for_moment)
                            g_tilde = D_t * nesterov
                        else:
                            D_t = None
                            g_tilde = nesterov

                        if g_tilde.norm() < 1e-8:
                            continue

                        # Group by shape for batched processing
                        m, n = nesterov.shape
                        shape_key = (min(m, n), max(m, n))
                        muon_shape_groups.setdefault(shape_key, []).append(
                            (group, p, g_tilde, p_star, orig_shape, D_t, state)
                        )

        # Pass 2: Batched Schatten updates
        for shape_key, items in muon_shape_groups.items():
            if len(items) == 1:
                group, param, g_tilde, p_star, orig_shape, D_t, state = items[0]

                update, _scale = schatten_update_batched(
                    g_tilde, p_star, pmax=self._pmax, normalize=self._normalize
                )

                # Symmetric preconditioning: map-out
                if D_t is not None:
                    update = D_t * update

                # Kappa-pin to spectrum-independent target Frobenius norm.
                m_u, n_u = update.size(-2), update.size(-1)
                target_fro = self._target_frobenius(group, m_u, n_u, p_star)
                update = update * (
                    target_fro / (update.norm(dim=(-2, -1), keepdim=True) + 1e-8)
                )

                effective_lr = group["lr"]

                # Weight decay and parameter update
                if group["weight_decay"] > 0:
                    param.mul_(1 - effective_lr * group["weight_decay"])

                param.add_(update.reshape(orig_shape), alpha=-effective_lr)

            else:
                # Batched processing
                g_tilde_list, p_list = [], []
                for item in items:
                    g_t = item[2]  # g_tilde
                    if g_t.size(-2) > g_t.size(-1):
                        g_t = g_t.transpose(-2, -1)
                    g_tilde_list.append(g_t)
                    p_list.append(item[3])

                g_tilde_batch = torch.stack(g_tilde_list)
                p_batch = torch.tensor(
                    p_list, device=g_tilde_batch.device, dtype=torch.float32
                )

                update_batch, _scale_batch = schatten_update_batched(
                    g_tilde_batch, p_batch, normalize=self._normalize
                )

                for i, item in enumerate(items):
                    group, param, g_tilde, p_star, orig_shape, D_t, state = item
                    update = update_batch[i]

                    if g_tilde.size(-2) > g_tilde.size(-1):
                        update = update.transpose(-2, -1)

                    # Symmetric preconditioning: map-out
                    if D_t is not None:
                        update = D_t * update

                    m_u, n_u = update.size(-2), update.size(-1)
                    target_fro = self._target_frobenius(group, m_u, n_u, p_star)
                    update = update * (target_fro / (update.norm(dim=(-2, -1)) + 1e-8))

                    effective_lr = group["lr"]

                    if group["weight_decay"] > 0:
                        param.mul_(1 - effective_lr * group["weight_decay"])

                    param.add_(update.reshape(orig_shape), alpha=-effective_lr)

        # Pass 3: All-Gather sync for distributed training
        if dist.is_initialized():
            for group in self.param_groups:
                params = group["params"]
                cache_key = id(group)
                params_pad = params + self._padding_cache[cache_key]
                for base_i in range(len(params))[::world_size]:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )

        return loss


class SingleDeviceSMuonWithAuxAdam(_KappaPinMixin, torch.optim.Optimizer):
    """
    Single-device Schatten-Muon optimizer. Same kappa-pinning logic as the
    distributed variant, without the all-gather plumbing.
    """

    def __init__(
        self,
        param_groups,
        param_names=None,
        pmin=1.02,
        pmax=10.0,
        eps=1e-5,
        init_p="pmax",
        p_method="exact_momentum",
        subsampling_ratio=0.1,
        moment_type="none",
        use_bias_correction=True,
        kappa_pin=True,
        kappa_adam_regime="noisy",
    ):
        self._param_names = param_names or {}
        self._pmin = pmin
        self._pmax = pmax
        self._init_p = {
            "pmax": pmax,
            "pmin": pmin,
        }[init_p]
        self._p_method = p_method
        self._subsampling_ratio = subsampling_ratio
        self._eps = eps
        self._normalize = False if moment_type != "none" else True

        if moment_type != "none" and moment_type not in MOMENT_REGISTRY:
            raise ValueError(
                f"Unknown moment_type '{moment_type}'. "
                f"Choose from: {list(MOMENT_REGISTRY.keys()) + ['none']}"
            )

        self._moment_type = moment_type
        self._is_parametric = is_parametric(moment_type)
        self._use_bias_correction = use_bias_correction

        self._kappa_enabled = bool(kappa_pin)
        if kappa_adam_regime not in ("noisy", "determ"):
            raise ValueError(
                f"kappa_adam_regime must be 'noisy' or 'determ', got {kappa_adam_regime!r}"
            )
        self._kappa_adam_regime = kappa_adam_regime

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

    def _get_or_create_moment(self, param, group):
        state = self.state[param]
        if "moment" not in state:
            if self._moment_type == "none":
                state["moment"] = None
            else:
                p_star = state.get("p_star", self._pmin)
                state["moment"] = create_moment(
                    self._moment_type,
                    p=p_star if self._is_parametric else None,
                    beta2=group["beta2"],
                    eps=group["eps"],
                    use_bias_correction=self._use_bias_correction,
                    p_threshold=0.99 * self._pmax,
                )
        return state["moment"]

    def get_p_state_for_logging(self):
        """Return {param_name: {"p_star": float}} for all Muon params."""
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
                    }
        return out

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

                # Handle Gram matrices vs raw activations
                if use_gram:
                    act_2d = act.float()
                else:
                    if "tightness" in self._p_method:
                        act_2d = act
                    else:
                        act_2d = act.reshape(act.size(0), -1) if act.ndim > 2 else act

                if torch.isnan(grad_2d).any() or torch.isnan(act_2d).any():
                    continue

                # Get or create P-approximator
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

                # Prepare method-specific kwargs
                kwargs = {}
                if any([m in self._p_method for m in ["momentum", "tightness"]]):
                    mom_buf = state.get("momentum_buffer", torch.zeros_like(grad))
                    muon_step = state.get("muon_step", 0) + 1
                    bc = 1.0 - group["momentum"] ** muon_step
                    mom_buf = mom_buf / bc
                    mom_2d = mom_buf.reshape(mom_buf.size(0), -1)
                    kwargs["mom_2d"] = mom_2d
                    kwargs["beta1"] = group["momentum"]

                # Compute p* with correct use_gram flag
                if "tightness" in self._p_method:
                    p_star, _ = approximator.update_and_compute_p(
                        state,
                        grad_2d,
                        act_2d,
                        use_gram=use_gram,
                        nesterov=True,
                        **kwargs,
                    )
                else:
                    p_star = approximator.update_and_compute_p(
                        state,
                        grad_2d,
                        act_2d,
                        use_gram=use_gram,
                        nesterov=True,
                        **kwargs,
                    )

                if p_star is None:
                    continue

                state["p_star"] = p_star

        if activations is not None:
            activations.clear()

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

            # Initialize state
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param)
                state["muon_step"] = 0
                state["p_star"] = self._init_p

            state["muon_step"] += 1
            step = state["muon_step"]

            # Get p_star
            p_star = state.get("p_star", self._pmin)

            # Nesterov momentum with bias correction
            mom_buf = state["momentum_buffer"]
            mom_buf.lerp_(grad, 1 - beta1)
            bc = 1.0 - beta1**step
            mom_buf_corrected = mom_buf / bc
            nesterov = mom_buf_corrected.lerp(grad, 1 - beta1)

            # Reshape to 2D
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

            # Schatten update using polar decomposition
            update, _scale = schatten_update_batched(g_tilde, p_star, pmax=self._pmax, normalize=self._normalize)

            # Symmetric preconditioning: map-out
            if D_t is not None:
                update = D_t * update

            # Kappa-pin: ||Delta W||_F = sqrt(min(m,n)) * kappa^(1/p)
            m_u, n_u = update.size(-2), update.size(-1)
            target_fro = self._target_frobenius(group, m_u, n_u, p_star)
            update = update * (target_fro / (update.norm() + 1e-8))

            effective_lr = group["lr"]

            # Weight decay and parameter update
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
