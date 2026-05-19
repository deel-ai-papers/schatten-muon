"""
Fixed Schatten-Muon optimizer with p=2 or p=4.

Provides two optimizer variants:
- mSGD: Momentum SGD with Schatten-p update (no second moment)
- AdamP: Adam-style with Schatten-p update (with second moment preconditioning)
"""

import torch

from smuon.p_root import fractional_power_from_polar
from smuon.moments import create_moment


def schatten_update_batched(
    G,
    p=2.0,
    epsilon=1e-3,
    max_extra_iters=5,
    linf_tol=5e-3,
    lambda_reg=1e-4,
    pmax=10.0,
):
    """
    Computes U·Σ^{1/p}·V^T using fractional_power_from_polar with a fixed p value.

    Args:
        G: Gradient tensor. Can be:
           - 2D: single matrix [m, n]
           - 3D: batch of matrices [batch, m, n]
        p: Schatten-p norm exponent (default: 2.0)
        epsilon: Regularization for Taylor expansion
        max_extra_iters: Maximum polynomial order
        linf_tol: L-infinity error tolerance for polynomial approximation
        lambda_reg: Regularization parameter
        pmax: Maximum p value for thresholding

    Returns:
        Update tensor of same shape as G
    """
    is_single = G.ndim == 2
    original_dtype = G.dtype

    # Add batch dimension if needed
    if is_single:
        G = G.unsqueeze(0)

    # Handle transposition for tall matrices
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        G = G.transpose(-2, -1)

    # Convert p to p_order (fractional_power_from_polar expects p in [0, 1])
    p_order = 1.0 / p

    # Create p_orders tensor for batched computation
    p_orders = torch.full((G.size(0),), p_order, device=G.device, dtype=torch.float32)

    # Fully vectorized execution
    output, _ = fractional_power_from_polar(
        G,
        p_orders,
        eigen_floor=epsilon,
        max_extra_iters=max_extra_iters,
        linf_tol=linf_tol,
        lambda_reg=lambda_reg,
        p_threshold=1.01 * (1 / pmax),
    )

    # Transpose back if needed
    if transposed:
        output = output.transpose(-2, -1)

    # Remove batch dimension if input was single
    if is_single:
        output = output.squeeze(0)

    return output.to(original_dtype)


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Standard Adam update."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class _BaseFixedSchattenOptimizer(torch.optim.Optimizer):
    """
    Base class for fixed Schatten-p optimizers.

    Supports both momentum-only (mSGD) and Adam-style preconditioning (AdamP).
    """

    def __init__(
        self,
        param_groups,
        param_names=None,
        p=2.0,
        moment_type="none",
        use_bias_correction=True,
    ):
        """
        Initialize base Schatten optimizer.

        Args:
            param_groups: Parameter groups with 'use_muon' flag
            param_names: Optional mapping from parameters to names (for logging)
            p: Fixed Schatten-p norm exponent (default: 2.0, can be 2 or 4)
            moment_type: Type of second moment ("none" or "adam")
            use_bias_correction: Whether to use bias correction for moments (default: True)
        """
        if p not in [2.0, 4.0]:
            raise ValueError(f"p must be 2.0 or 4.0, got {p}")

        if moment_type not in ["none", "adam"]:
            raise ValueError(
                f"moment_type must be 'none' or 'adam', got '{moment_type}'"
            )

        self._param_names = param_names or {}
        self._p = float(p)
        self._moment_type = moment_type
        self._use_bias_correction = use_bias_correction

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
                # Adam moment parameters (only used if moment_type="adam")
                group.setdefault("beta2", 0.999)
                group.setdefault("eps", 1e-8)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("lr_multiplier", 1.0)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)

        super().__init__(param_groups, dict())

    def _get_or_create_moment(self, param, group):
        """Get or create moment tracker for a parameter."""
        state = self.state[param]
        if "moment" not in state:
            if self._moment_type == "none":
                state["moment"] = None
            else:
                state["moment"] = create_moment(
                    self._moment_type,
                    p=None,  # Non-parametric Adam
                    beta2=group["beta2"],
                    eps=group["eps"],
                    use_bias_correction=self._use_bias_correction,
                )
        return state["moment"]

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

            state["muon_step"] += 1
            step = state["muon_step"]

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

            # Get moment tracker
            moment = self._get_or_create_moment(param, group)

            # Apply second moment preconditioning (if enabled)
            if moment is not None:
                # Update variance on the RAW gradient (following Adam convention)
                grad_2d_for_moment = (
                    grad.reshape(grad.size(0), -1) if grad.ndim > 2 else grad
                )
                D_t = moment.update(state, grad_2d_for_moment)
                # Apply preconditioning to Nesterov-corrected momentum
                g_tilde = D_t * nesterov
            else:
                D_t = None
                g_tilde = nesterov

            if g_tilde.norm() < 1e-8:
                continue

            # Schatten update using polar decomposition with fixed p
            update = schatten_update_batched(g_tilde, p=self._p)

            # Symmetric preconditioning: map-out (if moment was applied)
            if D_t is not None:
                update = D_t * update

            # Muon's tall/fat rescaling
            update = update * (max(1, update.size(-2) / update.size(-1)) ** 0.5)

            # Weight decay and parameter update
            if group["weight_decay"] > 0:
                param.mul_(1 - group["lr"] * group["weight_decay"])

            param.add_(update.reshape(orig_shape), alpha=-group["lr"])

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


class mSGD(_BaseFixedSchattenOptimizer):
    """
    Momentum SGD with fixed Schatten-p update (no second moment preconditioning).

    This optimizer uses:
    - Fixed p (2 or 4) for Schatten-p norm
    - Nesterov momentum
    - No second moment preconditioning (moment_type="none")

    Suitable for baseline comparisons or when you want SGD-style behavior
    with Schatten updates.
    """

    def __init__(
        self,
        param_groups,
        param_names=None,
        p=2.0,
    ):
        """
        Initialize mSGD optimizer.

        Args:
            param_groups: Parameter groups with 'use_muon' flag
            param_names: Optional mapping from parameters to names (for logging)
            p: Fixed Schatten-p norm exponent (2.0 or 4.0, default: 2.0)
        """
        super().__init__(
            param_groups=param_groups,
            param_names=param_names,
            p=p,
            moment_type="none",
            use_bias_correction=False,  # Not needed for mSGD
        )


class AdamP(_BaseFixedSchattenOptimizer):
    """
    Adam-style optimizer with fixed Schatten-p update and second moment preconditioning.

    This optimizer uses:
    - Fixed p (2 or 4) for Schatten-p norm
    - Nesterov momentum
    - Adam-style second moment preconditioning (moment_type="adam")

    The second moment is non-parametric (fixed exponent -0.25, independent of p).
    """

    def __init__(
        self,
        param_groups,
        param_names=None,
        p=2.0,
        use_bias_correction=True,
    ):
        """
        Initialize AdamP optimizer.

        Args:
            param_groups: Parameter groups with 'use_muon' flag
            param_names: Optional mapping from parameters to names (for logging)
            p: Fixed Schatten-p norm exponent (2.0 or 4.0, default: 2.0)
            use_bias_correction: Whether to use bias correction for Adam moment
                                 (default: True)
        """
        super().__init__(
            param_groups=param_groups,
            param_names=param_names,
            p=p,
            moment_type="adam",
            use_bias_correction=use_bias_correction,
        )
