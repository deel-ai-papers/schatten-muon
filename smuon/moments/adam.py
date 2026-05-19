import torch
from smuon.moments.base import SecondOrderMoment


class AdamMoment(SecondOrderMoment):
    """
    Adam-style diagonal variance tracking with exponential moving average.
    Returns V^{-1/4} or V^{-1/(4p)} when p is set to interpolate between standard Adam and Muon.

    With bias correction enabled (default), applies correction factor 1/(1-beta2^t)
    to the raw EMA before raising to power.
    """

    STATE_KEY = "exp_avg_sq_muon"
    USES_EMA = True

    def update(self, state, grad):
        grad_2d = grad.view(grad.size(0), -1) if grad.ndim > 2 else grad
        grad_fp32 = grad_2d.float()

        if self.STATE_KEY not in state:
            state[self.STATE_KEY] = torch.zeros_like(grad_fp32, dtype=torch.float32)

        self._increment_step(state)
        state[self.STATE_KEY].lerp_(grad_fp32.square(), 1 - self.beta2)

        # Apply bias correction before raising to power
        bias_correction = self._get_bias_correction(state)
        v_corrected = state[self.STATE_KEY] * bias_correction

        result = torch.pow(v_corrected + self.eps, self._get_exponent())
        return result if self.p is None else result.to(grad_2d.dtype)
