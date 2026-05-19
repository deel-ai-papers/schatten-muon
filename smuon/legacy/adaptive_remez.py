"""
SMuon Single-Device Optimizer.

A Schatten-p norm optimizer that smoothly interpolates between Adam (p→1)
and Muon (p→∞) via layerwise adaptive p selection. Uses the D-norm
sandwich from the Preconditioned Norms paper (Theorem 1):

  update = D_t · NS(D_t · G)

where D_t = 1/(V̂^{1/(4p)} + ε) is a p-dependent element-wise preconditioner.

No dual scaling or Frobenius normalization is applied — the sandwich
structure provides the correct update scale at every p value:
  p ≈ 1:   D_t² ≈ 1/√V,  NS ≈ id  → update ∝ G/√V (Adam)
  p → ∞:   D_t → I,       NS → UV^T → update = UV^T (Muon)

The update magnitude naturally scales with gradient magnitude, so the
optimizer slows down near convergence — unlike Frobenius-normalized
variants which produce constant-size steps regardless of gradient scale.

Usage:
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters()
                            if p.ndim >= 2 and "embed" not in n]
    embed_params  = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params   = [model.lm_head.weight]

    adam_groups = [
        dict(params=head_params,   lr=0.22),
        dict(params=embed_params,  lr=0.6),
        dict(params=scalar_params, lr=0.04),
    ]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False)
                  for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95,
                      use_muon=True)

    optimizer = SMuonWithAuxAdamSingleDevice([*adam_groups, muon_group])

    # In training loop (every N_p steps):
    optimizer.update_p_state(activations)
    optimizer.step()
"""

import torch
import warnings
import math

from smuon.svs.exact import get_full_svs, optimal_p
from smuon.svs.p_registry import create_p_approximator
from smuon.coeffs import COEFF_METHODS, _DEFAULT_L2_TOL, _DEFAULT_LINF_TOL
from smuon.coeffs.polar_express import optimal_composition
from smuon.moments import create_moment, is_parametric, MOMENT_REGISTRY

NS_COEFFS = optimal_composition(
    l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.03
)


# ---------------------------------------------------------------------------
# Newton-Schulz
# ---------------------------------------------------------------------------


def schatten_via_newtonschulz(G, coeffs, p):
    """
    Computes U·Σ^{1/p}·V^T via Newton-Schulz iteration.

    Always unscales: the Gelfand normalization factor s is removed
    so the output is the true Schatten-p steepest descent direction.
    """
    assert G.ndim == 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Check if there are any non-trivial NS iterations (not just identity padding)
    has_ns_iters = any(abs(b) > 1e-9 or abs(c) > 1e-9 for a, b, c in coeffs)

    cpt = 1
    for a, b, c in coeffs:
        A = X @ X.mT
        if cpt == 1 and has_ns_iters:
            s = torch.rsqrt(
                torch.clamp_min(torch.linalg.norm(A, ord="fro", dim=(-2, -1)), 1e-7)
            )
            X = X * s
            A = A * s**2
        B = b * A + c * A @ A
        X = a * X + B @ X
        cpt += 1

    if has_ns_iters:
        X = X * (1.0 / s) ** (1.0 / p)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# ---------------------------------------------------------------------------
# Adam update (for non-Muon parameters)
# ---------------------------------------------------------------------------


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class SMuonWithAuxAdamSingleDevice(torch.optim.Optimizer):
    """
    Schatten-Muon optimizer for single-device training.

    Applies Schatten-p spectral updates (with adaptive per-layer p) to
    matrix parameters, and standard AdamW to the rest. Uses configurable
    moment preconditioners with optional grafting.

    Parameters
    ----------
    param_groups : list[dict]
        Each group must have ``use_muon=True`` or ``use_muon=False``.
        Muon groups accept: lr, momentum, beta2, eps, weight_decay.
        Adam groups accept: lr, lr_multiplier, betas, eps, weight_decay.
    param_names : dict[Parameter, str], optional
        Maps parameters to names for logging.
    coeff_method : str
        Coefficient solver from COEFF_METHODS (default: "safe_lagrangian").
    moment_type : str
        Moment preconditioner type. Options:
        - "adam", "adagrad", "adafactor", "sania": Fixed exponent (-1/4 or -1/2 for sania)
        - "padam", "padagrad", "padafactor", "psania": Parametric, scales with p_star
        - "none": No preconditioning (pure Muon/NS)
    pmin : float
        Minimum Schatten exponent (default: 1.02).
    pmax : float
        Maximum Schatten exponent (default: 35.0).
    p_method : str
        Method for computing optimal p. Options: "exact" (full SVD), "approx"
        (subsampled SVD with bounded tail correction), "approx_momentum"
        (double approx using momentum), or "exact_momentum" (exact with momentum).
    subsampling_ratio : float
        Fraction of singular values to compute when p_method="approx" or "approx_momentum".
        k = int(ratio * min(m, n)). Default: 0.1 (10%).
    use_bias_correction : bool
        Whether to apply bias correction for EMA-based moments (Adam/Adafactor/Sania).
    grafting : bool
        Whether to enable grafting (scale matching to reference optimizer).
    graft_exp : float
        Exponent for grafting reference (-0.5 for Adam-style scaling).
    """

    def __init__(
        self,
        param_groups,
        param_names=None,
        coeff_method="linf_remez_adaptive",
        moment_type="padam",
        pmin=1.02,
        pmax=35.0,
        eps=1e-5,
        init_p="pmax",
        p_method="approx_momentum",
        subsampling_ratio=0.1,
        use_bias_correction=True,
        grafting=True,
        graft_exp=-0.5,
        save_distributions=False,
        save_dir=None,
        weights_to_save=None,
    ):
        if coeff_method not in COEFF_METHODS:
            raise ValueError(
                f"Unknown coeff_method '{coeff_method}'. "
                f"Choose from: {list(COEFF_METHODS.keys())}"
            )
        if moment_type != "none" and moment_type not in MOMENT_REGISTRY:
            raise ValueError(
                f"Unknown moment_type '{moment_type}'. "
                f"Choose from: {list(MOMENT_REGISTRY.keys()) + ['none']}"
            )
        if p_method not in ("exact", "approx", "approx_momentum", "exact_momentum"):
            raise ValueError(
                f"Unknown p_method '{p_method}'. Choose from: 'exact', 'approx', 'approx_momentum', 'exact_momentum'"
            )
        self._coeff_method = COEFF_METHODS[coeff_method]
        self._coeff_method_name = coeff_method
        self._param_names = param_names or {}
        self._pmin = pmin
        self._pmax = pmax
        self._init_p = {
            "pmax": pmax,
            "pmin": pmin,
        }[init_p]
        self._p_method = p_method
        self._subsampling_ratio = subsampling_ratio
        self._moment_type = moment_type
        self._is_parametric = is_parametric(moment_type)
        self._use_bias_correction = use_bias_correction
        self._grafting = grafting
        self._graft_exp = graft_exp
        self._save_distributions = save_distributions
        self._save_dir = save_dir
        self._weights_to_save = weights_to_save

        # Warn if grafting is enabled but moment_type is "none"
        if self._grafting and self._moment_type == "none":
            warnings.warn(
                "Grafting is enabled but moment_type='none'. Grafting will be disabled "
                "because it requires a moment preconditioner to compute the reference scale. "
                "Set moment_type to 'adam', 'padam', or another preconditioner to enable grafting.",
                UserWarning,
                stacklevel=2,
            )

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

        self._eps = eps
        super().__init__(param_groups, dict())

    def get_p_state_for_logging(self):
        """Return {param_name: {"p_star": float, "coeffs": list}} for all Muon params."""
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
                        "coeffs": state.get("coeffs"),
                    }
        return out

    def _get_or_create_moment(self, param, group):
        """Get or create a moment instance for a parameter."""
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
                )
        return state["moment"]

    @torch.no_grad()
    def update_p_state(self, activations=None, global_step=None):
        """
        Update optimal p and NS coefficients for each Muon parameter.

        Parameters
        ----------
        activations : dict[Parameter, Tensor] or None
            Maps parameters to their input activations. If a parameter
            has no activation recorded, it keeps its current p (or
            defaults to pmin on first call, giving Adam-like warmup).
        global_step : int or None
            Current training step number (used for saving distributions).
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
                    grad_2d = grad.view(grad.size(0), -1)
                elif grad.ndim == 2:
                    grad_2d = grad
                else:
                    continue

                if activations is None or p not in activations:
                    if "p_star" not in state:
                        state["p_star"] = self._init_p
                        p_order = 1.0 / state["p_star"]
                        coeffs = NS_COEFFS if state["p_star"] == self._pmax else None
                        state["coeffs"] = (
                            [(c[0], c[1], c[2]) for c in coeffs]
                            if coeffs is not None
                            else None
                        )
                    continue

                act = activations.pop(p)
                act_2d = act.view(act.size(0), -1) if act.ndim > 2 else act

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
                if self._p_method in ("approx_momentum", "exact_momentum"):
                    mom_buf = state.get("momentum_buffer", torch.zeros_like(grad))
                    mom_2d = (
                        mom_buf.view(mom_buf.size(0), -1)
                        if mom_buf.ndim > 2
                        else mom_buf
                    )
                    kwargs["mom_2d"] = mom_2d
                    kwargs["beta1"] = group["momentum"]

                    # Pass original shapes for saving
                    if self._save_distributions and self._p_method == "exact_momentum":
                        kwargs["grad_orig"] = grad
                        kwargs["act_orig"] = act
                        kwargs["mom_orig"] = mom_buf

                        # Generate save path with global step
                        if self._save_dir is not None:
                            import os

                            param_name = self._param_names.get(p, f"param_{id(p)}")

                            # Check if this parameter should be saved
                            should_save = True
                            if self._weights_to_save is not None:
                                should_save = any(
                                    key in param_name for key in self._weights_to_save
                                )

                            if should_save:
                                # Sanitize parameter name for filesystem
                                param_name = param_name.replace(".", "_").replace(
                                    "/", "_"
                                )
                                step_str = (
                                    f"step_{global_step:03d}"
                                    if global_step is not None
                                    else "step_unknown"
                                )
                                save_path = os.path.join(
                                    self._save_dir, f"{step_str}_{param_name}.pth"
                                )
                                kwargs["save_path"] = save_path

                # Compute p* (use_gram=False for single device)
                p_star = approximator.update_and_compute_p(
                    state, grad_2d, act_2d, use_gram=False, **kwargs
                )

                if p_star is None:
                    continue

                state["p_star"] = p_star

                p_order = 1.0 / p_star

                # Call coefficient method with appropriate parameters
                if "adaptive" in self._coeff_method_name:
                    tol = (
                        _DEFAULT_L2_TOL
                        if "l2_lagrangian" in self._coeff_method_name
                        else _DEFAULT_LINF_TOL
                    )
                    coeffs = self._coeff_method(p_order=p_order, tol=tol, max_steps=5)
                else:
                    coeffs = self._coeff_method(num_steps=5, p_order=p_order)

                state["coeffs"] = (
                    [(c[0], c[1], c[2]) for c in coeffs] if coeffs is not None else None
                )

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

            # ---- init ----
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param)
                state["muon_step"] = 0

            state["muon_step"] += 1
            step = state["muon_step"]

            # Get p_star and coeffs
            p_star = state.get("p_star", self._pmin)
            coeffs = state.get("coeffs", None)

            # Step 1: Nesterov momentum with bias correction
            mom_buf = state["momentum_buffer"]
            mom_buf.lerp_(grad, 1 - beta1)
            # Nesterov lookahead: beta1 * mom_buf + (1 - beta1) * grad
            nesterov = mom_buf.lerp(grad, 1 - beta1)
            # Bias correction
            bc = 1.0 - beta1**step
            nesterov = nesterov / bc

            # Step 2: Reshape for 2D operations
            orig_shape = nesterov.shape
            if nesterov.ndim > 2:
                nesterov = nesterov.view(nesterov.size(0), -1)

            # Step 3: Update moment with momentum-smoothed signal (2D)
            # This tracks variance of the actual quantity being preconditioned,
            # not the raw gradient which has different scale characteristics.
            moment = self._get_or_create_moment(param, group)
            if moment is not None and self._is_parametric:
                moment.p = p_star

            if moment is not None:
                D_t = moment.update(state, nesterov)
            else:
                D_t = None

            # Step 4: Map-in preconditioning
            if D_t is not None:
                g_tilde = D_t * nesterov
            else:
                g_tilde = nesterov

            # Grafting: compute reference optimizer's update Frobenius norm
            if self._grafting and D_t is not None:
                moment_exp = moment._get_exponent()
                if moment_exp != 0.0:
                    ratio = self._graft_exp / moment_exp
                    scale_ref = (nesterov * D_t.pow(ratio)).norm() + 1e-8
                else:
                    scale_ref = None
            else:
                scale_ref = None

            # Step 5: Newton-Schulz projection
            if coeffs is not None:
                delta_ns = schatten_via_newtonschulz(g_tilde, coeffs, p_star)
            else:
                if scale_ref is not None:
                    update = D_t * g_tilde if D_t is not None else g_tilde
                    scale_update = update.norm() + 1e-8
                    update = update * (scale_ref / scale_update)
                else:
                    update = D_t * g_tilde if D_t is not None else g_tilde
                    update = update / (update.norm() + 1e-8)

                if group["weight_decay"] > 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update.view(orig_shape), alpha=-group["lr"])
                continue

            # Step 6: Map-out: apply D_t after NS projection
            if D_t is not None:
                update = D_t * delta_ns
            else:
                update = delta_ns

            # Step 7: Apply grafting or normalization
            if scale_ref is not None:
                scale_update = update.norm() + 1e-8
                update = update * (scale_ref / scale_update)
            else:
                p_norm = 2 if p_star < 2 else p_star
                update = update / (update.norm(p=p_norm) + 1e-8)

            # Step 8: Weight decay and parameter update
            if group["weight_decay"] > 0:
                param.mul_(1 - group["lr"] * group["weight_decay"])

            param.add_(update.view(orig_shape), alpha=-group["lr"])

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
