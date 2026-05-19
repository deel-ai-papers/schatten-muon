"""
Baseline Muon optimizers without adaptive p selection.

These optimizers use p=∞ (standard Muon) without computing optimal p values.
Useful as baselines for comparison with adaptive SMuon variants.
"""

import torch
import torch.distributed as dist

from smuon.coeffs.polar_express import optimal_composition

NS_COEFFS = optimal_composition(
    l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.03
)


def zeropower_via_newtonschulz5(G):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng

    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in NS_COEFFS:
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Standard Adam update."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Baseline Muon optimizer (p=∞) for multi-GPU training.

    Standard Muon without adaptive p selection. Uses Newton-Schulz orthogonalization
    via polar decomposition (p→∞ limit of Schatten-p norm).

    Parameters
    ----------
    param_groups : list
        Parameter groups with 'use_muon' key to distinguish Muon vs Adam params.
    """

    def __init__(self, param_groups):
        self._padding_cache = {}

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0)
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

                        state["muon_step"] += 1
                        step = state["muon_step"]

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

                        # Group by shape for batched processing
                        m, n = nesterov.shape
                        shape_key = (min(m, n), max(m, n))
                        muon_shape_groups.setdefault(shape_key, []).append(
                            (group, p, nesterov, orig_shape)
                        )

        # Pass 2: Batched Muon updates (p=∞)
        for shape_key, items in muon_shape_groups.items():
            if len(items) == 1:
                group, param, nesterov, orig_shape = items[0]

                # Muon update (orthogonalization, p=∞)
                update = zeropower_via_newtonschulz5(nesterov)
                update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

                # Weight decay and parameter update
                if group["weight_decay"] > 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                param.add_(update.reshape(orig_shape), alpha=-group["lr"])
            else:
                # Batched processing
                nesterov_list = []
                for item in items:
                    nesterov = item[2]
                    # Transpose tall matrices
                    if nesterov.size(-2) > nesterov.size(-1):
                        nesterov = nesterov.transpose(-2, -1)
                    nesterov_list.append(nesterov)

                # Stack into batch
                nesterov_batch = torch.stack(nesterov_list)

                # Batched Muon update
                update_batch = zeropower_via_newtonschulz5(nesterov_batch)

                # Apply updates
                for i, item in enumerate(items):
                    group, param, nesterov, orig_shape = item
                    update = update_batch[i]

                    # Transpose back if needed
                    if nesterov.size(-2) > nesterov.size(-1):
                        update = update.transpose(-2, -1)

                    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

                    # Weight decay and parameter update
                    if group["weight_decay"] > 0:
                        param.mul_(1 - group["lr"] * group["weight_decay"])

                    param.add_(update.reshape(orig_shape), alpha=-group["lr"])

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


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Baseline Muon optimizer (p=∞) for single-device training.

    Standard Muon without adaptive p selection. Uses Newton-Schulz orthogonalization
    via polar decomposition (p→∞ limit of Schatten-p norm).

    Parameters
    ----------
    param_groups : list
        Parameter groups with 'use_muon' key to distinguish Muon vs Adam params.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0)
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

            # Bias correction
            bc = 1.0 - beta1**step
            mom_buf_corrected = mom_buf / bc

            # Nesterov lookahead
            nesterov = mom_buf_corrected.lerp(grad, 1 - beta1)

            # Reshape to 2D
            orig_shape = nesterov.shape
            if nesterov.ndim > 2:
                nesterov = nesterov.reshape(nesterov.size(0), -1)

            # Muon update (orthogonalization, p=∞)
            update = zeropower_via_newtonschulz5(nesterov)
            update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

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
