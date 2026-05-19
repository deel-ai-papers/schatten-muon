import torch
from smuon.moments.base import SecondOrderMoment


class AdafactorMoment(SecondOrderMoment):
    """
    Memory-efficient factored second moment tracking.
    Decomposes variance into row and column factors.
    Returns V^{-1/4} or V^{-1/(4p)} when p is set.

    With bias correction enabled (default), applies correction factor 1/(1-beta2^t)
    to the factored variance before raising to power.
    """

    USES_EMA = True

    def update(self, state, grad):
        grad_2d = grad.view(grad.size(0), -1) if grad.ndim > 2 else grad
        grad_fp32 = grad_2d.float()

        if "row_var" not in state:
            state["row_var"] = torch.zeros(
                grad_fp32.shape[0], device=grad_fp32.device, dtype=torch.float32
            )
            state["col_var"] = torch.zeros(
                grad_fp32.shape[1], device=grad_fp32.device, dtype=torch.float32
            )

        self._increment_step(state)
        grad_sq = grad_fp32.square()
        state["row_var"].lerp_(grad_sq.mean(dim=1), 1 - self.beta2)
        state["col_var"].lerp_(grad_sq.mean(dim=0), 1 - self.beta2)

        # Apply bias correction to the factors
        bias_correction = self._get_bias_correction(state)
        row_var_corrected = state["row_var"] * bias_correction
        col_var_corrected = state["col_var"] * bias_correction

        row_mean = row_var_corrected.mean().clamp_min_(self.eps)
        r_factor = (row_var_corrected / row_mean).unsqueeze(1)
        c_factor = col_var_corrected.unsqueeze(0)
        v_approx = r_factor * c_factor

        result = torch.pow(v_approx + self.eps, self._get_exponent())
        return result if self.p is None else result.to(grad_2d.dtype)
