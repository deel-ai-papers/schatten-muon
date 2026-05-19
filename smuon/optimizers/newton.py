# smuon/optimizers/newton_smuon.py
import torch
from smuon.optimizers.exact import _MuonBaseWithTracking, _exact_schatten_update
from smuon.optimizers.baseline import zeropower_via_newtonschulz5, NS_COEFFS


# ---------------------------------------------------------------------
# K_inv computation helpers
# ---------------------------------------------------------------------
def _chol_inv_full(
    K: torch.Tensor, gamma_rel: float, eps: float = 1e-8
) -> torch.Tensor:
    """Damped Cholesky inverse of K (n, n). Returns (K + gamma I)^-1."""
    n = K.size(-1)
    gamma = gamma_rel * K.diagonal(dim1=-2, dim2=-1).mean() + eps
    eye = torch.eye(n, device=K.device, dtype=K.dtype)
    Kd = K + gamma * eye
    L, info = torch.linalg.cholesky_ex(Kd, upper=False, check_errors=False)
    if int(info.item() if info.ndim == 0 else info.max().item()) != 0:
        # Fallback: identity. Caller should log this.
        return eye.clone()
    return torch.cholesky_inverse(L, upper=False)


def _chol_inv_blocks(
    K_blocks: torch.Tensor, gamma_rel: float, eps: float = 1e-8
) -> torch.Tensor:
    """
    Batched damped Cholesky inverse over a stack of (k, d, d) SPD matrices.
    Mirrors the speedrun's batched Cholesky-inverse refresh.
    """
    k, d, _ = K_blocks.shape
    diag = K_blocks.diagonal(dim1=-2, dim2=-1)
    gamma = gamma_rel * (diag.sum(dim=-1) / float(d)) + eps  # (k,)
    eye = torch.eye(d, device=K_blocks.device, dtype=K_blocks.dtype)
    Kd = K_blocks + gamma.view(k, 1, 1) * eye
    L, info = torch.linalg.cholesky_ex(Kd, upper=False, check_errors=False)
    inv = torch.cholesky_inverse(L, upper=False)
    if info.numel() == k:
        bad = info != 0
        if bad.any():
            inv_bad = eye.clone()
            inv[bad] = inv_bad
    return inv


def _gram_to_blocks(gram: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Extract block-diagonal d x d blocks from a (k*d, k*d) Gram matrix.
    Matches the Newton-Muon speedrun: for MLP c_proj, the input has dim 4d
    which we model as four independent d-dim sub-inputs.
    """
    n = gram.size(-1)
    assert n % block_size == 0, f"Gram dim {n} not divisible by block_size {block_size}"
    k = n // block_size
    blocks = torch.empty(
        (k, block_size, block_size), device=gram.device, dtype=gram.dtype
    )
    for j in range(k):
        s, e = j * block_size, (j + 1) * block_size
        blocks[j] = gram[s:e, s:e]
    return blocks


# ---------------------------------------------------------------------
# Base class with shared Newton-Muon machinery
# ---------------------------------------------------------------------
class _NewtonBase(_MuonBaseWithTracking):
    """
    Adds Newton-Muon right-preconditioning on top of _ExactSMuonBase.

    Per-param state:
        K_running     : running Gram (full or blocks)
        K_inv_spec    : dict consumed by the optimizer step and p* approximator
        K_block_size  : set for layers using block-diagonal inverse

    Extra hyperparameters:
        nm_K_beta       EWMA for running Gram                   (default 0.95)
        nm_gamma_rel    damping: gamma = gamma_rel * tr(K)/n    (default 0.2)
        nm_refresh_every  K_inv refresh cadence in optimizer steps (default 32)
    """

    def __init__(
        self,
        *args,
        nm_K_beta: float = 0.95,
        nm_gamma_rel: float = 0.2,
        nm_refresh_every: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._nm_K_beta = float(nm_K_beta)
        self._nm_gamma_rel = float(nm_gamma_rel)
        self._nm_refresh_every = int(nm_refresh_every)
        self._nm_global_step = 0

    # ------------------------------------------------------------------
    # K update. Call this every step with the recorder's activations dict.
    # The dict is expected to contain Gram matrices (use_gram=True on the
    # recorder side). For layers requiring block-diag inversion, pass a
    # `block_hint` dict: {param: block_size}.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_K_state(self, activations: dict, block_hint: dict | None = None):
        block_hint = block_hint or {}

        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                if p not in activations:
                    continue
                gram = activations[p].float()
                if gram.ndim != 2 or gram.size(0) != gram.size(1):
                    # Recorder wasn't in use_gram mode; fall back to A^T A.
                    A = gram.reshape(-1, gram.size(-1)) if gram.ndim > 2 else gram
                    gram = A.transpose(-2, -1) @ A
                    gram = gram / max(1, A.size(0))

                state = self.state[p]
                block_size = block_hint.get(p, None)

                if block_size is not None:
                    new_blocks = _gram_to_blocks(gram, block_size)
                    if (
                        "K_running" not in state
                        or state.get("K_block_size") != block_size
                    ):
                        state["K_running"] = new_blocks.clone()
                        state["K_block_size"] = block_size
                    else:
                        state["K_running"].mul_(self._nm_K_beta).add_(
                            new_blocks, alpha=(1.0 - self._nm_K_beta)
                        )
                else:
                    if (
                        "K_running" not in state
                        or state["K_running"].shape != gram.shape
                    ):
                        state["K_running"] = gram.clone()
                        state.pop("K_block_size", None)
                    else:
                        state["K_running"].mul_(self._nm_K_beta).add_(
                            gram, alpha=(1.0 - self._nm_K_beta)
                        )

    # ------------------------------------------------------------------
    # K_inv refresh. Call on refresh steps.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def refresh_K_inv(self):
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                state = self.state[p]
                if "K_running" not in state:
                    continue
                K = state["K_running"]
                block_size = state.get("K_block_size", None)
                if block_size is not None:
                    inv_blocks = _chol_inv_blocks(K, self._nm_gamma_rel)
                    state["K_inv_spec"] = {
                        "blocks": inv_blocks,
                        "block_size": block_size,
                    }
                else:
                    inv = _chol_inv_full(K, self._nm_gamma_rel)
                    state["K_inv_spec"] = {"full": inv}

    # ------------------------------------------------------------------
    # Right-precondition a (m, n) gradient by state's K_inv_spec.
    # Matches _apply_K_inv_right in newton_exact.py.
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_K_inv_right(M: torch.Tensor, spec: dict) -> torch.Tensor:
        if spec is None:
            return M
        if "full" in spec:
            return M @ spec["full"].to(M.dtype)
        if "blocks" in spec:
            blocks = spec["blocks"].to(M.dtype)
            d = int(spec["block_size"])
            k = blocks.size(0)
            m, n = M.shape
            assert n == k * d, f"M has {n} cols, expected {k*d}"
            M_view = M.view(m, k, d).permute(1, 0, 2).contiguous()  # (k, m, d)
            out = torch.bmm(M_view, blocks)  # (k, m, d)
            return out.permute(1, 0, 2).contiguous().view(m, n)
        return M

    # ------------------------------------------------------------------
    # Forward K_inv_spec into the p* approximator (only for SNewtonMuon;
    # NewtonMuon overrides update_p_state to be a no-op).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_p_state(self, activations=None, global_step=None, use_gram=False):
        # Update K first if activations are provided (gram matrices).
        if activations is not None:
            # Pass a copy because the base class pops entries from `activations`.
            self.update_K_state(dict(activations))

            # Refresh K_inv on cadence.
            step = int(global_step) if global_step is not None else self._nm_global_step
            if (step % self._nm_refresh_every) == 0:
                self.refresh_K_inv()

        # Now run the base class's p* update; the approximator will pull
        # K_inv_spec from state via the override below.
        super().update_p_state(
            activations=activations, global_step=global_step, use_gram=use_gram
        )

    # ------------------------------------------------------------------
    # Muon step with right-preconditioning on the Nesterov momentum.
    # Structure mirrors _ExactSMuonBase._step_muon; only g_tilde construction
    # and the Muon update body differ.
    # ------------------------------------------------------------------
    def _step_muon(self, group):
        beta1 = group["momentum"]
        self._nm_global_step += 1

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
            nesterov = (mom_buf / bc).lerp(grad, 1 - beta1)

            orig_shape = nesterov.shape
            if nesterov.ndim > 2:
                nesterov = nesterov.reshape(nesterov.size(0), -1)

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

            # Newton-Muon right-preconditioning.
            spec = state.get("K_inv_spec", None)
            if spec is not None:
                g_tilde = self._apply_K_inv_right(g_tilde, spec)

            if g_tilde.norm() < 1e-8:
                continue

            update = self._compute_muon_update(g_tilde, p_star)

            # Symmetric left-diagonal preconditioning, if active.
            if D_t is not None:
                update = D_t * update

            if self._match_muon_frobenius:
                m, n = update.size(-2), update.size(-1)
                cur_norm = update.norm()
                update = update * (float(min(m, n)) ** 0.5 / (cur_norm + 1e-8))

            update = update * (max(1, update.size(-2) / update.size(-1)) ** 0.5)

            if self._use_alpha_star:
                effective_lr = group["lr"] * self._effective_alpha_star(
                    state, raw_alpha_star
                )
            else:
                effective_lr = group["lr"]

            if group["weight_decay"] > 0:
                param.mul_(1 - effective_lr * group["weight_decay"])

            param.add_(update.reshape(orig_shape), alpha=-effective_lr)

    def _compute_muon_update(self, nesterov_2d, p_star):
        raise NotImplementedError


# ---------------------------------------------------------------------
# Public optimizers
# ---------------------------------------------------------------------
class NewtonMuon(_NewtonBase):
    """
    Pure Newton-Muon: msgn(G @ K_inv). Ignores p* and always applies UV^T.

    Useful as:
      - a sanity-check that the K_inv plumbing reproduces the NM paper's wins,
      - a baseline to separate the effect of K_inv alone from the effect of p*.
    """

    def __init__(self, *args, **kwargs):
        # Force pure Muon semantics: no moment, no alpha_star, init p = pmax.
        kwargs.setdefault("moment_type", "none")
        kwargs.setdefault("use_alpha_star", False)
        kwargs.setdefault("precond_aware_p", False)
        kwargs.setdefault("init_p", "pmax")
        super().__init__(*args, **kwargs)

    def _compute_muon_update(self, nesterov_2d, p_star):
        # Always the polar factor; p* is computed/logged but not used.
        return zeropower_via_newtonschulz5(nesterov_2d)

    @torch.no_grad()
    def update_p_state(self, activations=None, global_step=None, use_gram=False):
        # Still update K from activations and refresh K_inv, but skip p*.
        if activations is not None:
            self.update_K_state(dict(activations))
            step = int(global_step) if global_step is not None else self._nm_global_step
            if (step % self._nm_refresh_every) == 0:
                self.refresh_K_inv()
            activations.clear()
