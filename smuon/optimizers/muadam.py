"""
MuAdam optimizer: Muon with second-order moment on orthogonalized gradients.
"""

import torch

from smuon.coeffs.polar_express import optimal_composition

NS_COEFFS = optimal_composition(
    l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.03
)


def zeropower_via_newtonschulz5(G):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are selected to maximize the slope at zero.
    Returns something close to UV^T where USV^T = G is the SVD.
    """
    assert G.ndim >= 2

    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for a, b, c in NS_COEFFS:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Standard Adam update for auxiliary parameters."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.9)
                group.setdefault("beta2", 0.999)
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0)
                group.setdefault("use_bias_correction", True)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("lr_multiplier", 1.0)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)

        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muadam(group)
            else:
                self._step_adam(group)

        return loss

    def _step_muadam(self, group):
        beta1 = group["momentum"]
        beta2 = group["beta2"]
        eps = group["eps"]
        use_bias_correction = group["use_bias_correction"]

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

            # Step 1: Update first-order momentum normally
            mom_buf = state["momentum_buffer"]
            mom_buf.lerp_(grad, 1 - beta1)

            # Step 2: Update second-order moment
            if "exp_avg_sq_muon" not in state:
                state["exp_avg_sq_muon"] = torch.zeros_like(grad, dtype=torch.float32)

            v_t = state["exp_avg_sq_muon"]
            v_t.lerp_(grad.square(), 1 - beta2)

            # Bias correction for second moment
            bc2 = 1.0 - beta2**step if use_bias_correction else 1.0
            v_t_corrected = v_t / bc2
            D_t = torch.pow(v_t_corrected + eps, -0.25).to(grad.dtype)

            # Bias correction for first moment
            bc1 = 1.0 - beta1**step if use_bias_correction else 1.0
            mom_buf_corrected = mom_buf / bc1

            # Nesterov lookahead
            nesterov = mom_buf_corrected.lerp(grad, 1 - beta1)

            # Reshape to 2D
            orig_shape = nesterov.shape
            if nesterov.ndim > 2:
                nesterov = nesterov.reshape(nesterov.size(0), -1)
 
            # Apply Newton-Schulz orthogonalization to the sign with map-in
            update = zeropower_via_newtonschulz5(D_t * nesterov)
            # Map-out
            update = D_t * update

            # Weight decay and parameter update
            if group["weight_decay"] > 0:
                param.mul_(1 - group["lr"] * group["weight_decay"])

            param.add_(update.reshape(orig_shape), alpha=-group["lr"])

    def _step_adam(self, group):
        """Standard Adam update for auxiliary parameters."""
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
