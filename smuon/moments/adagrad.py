import torch
from smuon.moments.base import SecondOrderMoment


class AdaGradMoment(SecondOrderMoment):
    """
    AdaGrad-style cumulative squared gradient tracking (no decay).
    Returns G^{-1/4} or G^{-1/(4p)} when p is set.

    Note: AdaGrad uses cumulative sum, not EMA, so bias correction is not applicable.
    """

    STATE_KEY = "sum_sq_muon"
    USES_EMA = False  # Cumulative sum, no bias correction needed

    def update(self, state, grad):
        grad_2d = grad.view(grad.size(0), -1) if grad.ndim > 2 else grad
        grad_fp32 = grad_2d.float()

        if self.STATE_KEY not in state:
            state[self.STATE_KEY] = torch.zeros_like(grad_fp32, dtype=torch.float32)

        self._increment_step(state)
        state[self.STATE_KEY].add_(grad_fp32.square())

        result = torch.pow(state[self.STATE_KEY] + self.eps, self._get_exponent())
        return result if self.p is None else result.to(grad_2d.dtype)
