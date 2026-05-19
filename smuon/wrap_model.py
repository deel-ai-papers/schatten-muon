"""
Model wrapper for recording activations during forward pass.

Used by adaptive optimizers to compute optimal p values based on
both gradient and activation singular value distributions.

Supports two modes:
- Raw activations: Store full activation tensors (default, single GPU)
- Gram matrices: Store A^T A for distributed aggregation across GPUs
"""

import torch
import torch.nn as nn
from contextlib import contextmanager


class ActivationRecorder:
    """
    Wraps a model to record input activations for weight parameters.

    Activations are only recorded when inside a `recording()` context or
    when `record_activations` is explicitly set to True.

    Parameters
    ----------
    model : nn.Module
        The model to wrap.
    use_gram : bool
        If True, store Gram matrices (A^T A) instead of raw activations.
        Gram matrices can be summed across GPUs for distributed training:
        (A_1; A_2)^T (A_1; A_2) = A_1^T A_1 + A_2^T A_2

    Example
    -------
    >>> model = nn.Linear(10, 5)
    >>> recorder = ActivationRecorder(model, use_gram=True)
    >>> with recorder.recording():
    ...     out = model(torch.randn(32, 10))
    >>> grams = recorder.get_activations()  # Returns Gram matrices
    >>> recorder.clear()
    """

    def __init__(self, model: nn.Module, use_gram: bool = False):
        self.model = model
        self.use_gram = use_gram
        self.record_activations = False
        self._activations: dict[nn.Parameter, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all supported layers."""
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                handle = module.register_forward_hook(self._make_hook(module))
                self._hooks.append(handle)

    def _make_hook(self, module: nn.Module):
        """Create a hook function for the given module."""

        def hook(mod, inputs, output):
            if not self.record_activations:
                return
            if len(inputs) == 0:
                return

            inp = inputs[0]
            # Associate with the weight parameter
            if hasattr(mod, "weight") and mod.weight is not None:
                # Detach to avoid graph retention
                act = inp.detach()

                if self.use_gram:
                    act_2d = self._get_effective_activation(mod, act)
                    self._activations[mod.weight] = act_2d.transpose(-2, -1) @ act_2d
                else:
                    # Store an im2col'd 2D view for Conv layers so the feature axis
                    # (C * kH * kW) is explicit; Linear activations pass through unchanged
                    # and _orient_activation reshapes them downstream.
                    if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
                        self._activations[mod.weight] = self._get_effective_activation(
                            mod, act
                        )
                    else:
                        self._activations[mod.weight] = act

        return hook

    def _get_effective_activation(
        self, mod: nn.Module, act: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the effective 2D activation matrix for Gram computation.

        For Linear: (batch, in_features) -> (batch, in_features)
        For Conv: use unfold to get (batch * num_patches, C * kH * kW)
        """
        if isinstance(mod, nn.Linear):
            # Linear: flatten any extra dims, keep last as features
            return act.reshape(-1, act.size(-1)).float()

        elif isinstance(mod, nn.Conv2d):
            # Conv2d: unfold to get im2col-style patches
            # Input: (N, C, H, W)
            # Output: (N * num_patches, C * kH * kW)
            kH, kW = mod.kernel_size
            stride = mod.stride
            padding = mod.padding
            dilation = mod.dilation

            # Use unfold to extract patches
            # unfolded shape: (N, C * kH * kW, num_patches)
            unfolded = torch.nn.functional.unfold(
                act.float(),
                kernel_size=(kH, kW),
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            # Reshape to (N * num_patches, C * kH * kW)
            N, features, num_patches = unfolded.shape
            act_2d = unfolded.permute(0, 2, 1).reshape(N * num_patches, features)
            return act_2d

        elif isinstance(mod, nn.Conv1d):
            # Conv1d: similar to Conv2d but 1D
            kW = mod.kernel_size[0]
            stride = mod.stride[0]
            padding = mod.padding[0]
            dilation = mod.dilation[0]

            # unfold for 1D: (N, C, L) -> (N, C * kW, num_patches)
            unfolded = act.float().unfold(2, kW, stride)  # (N, C, num_patches, kW)
            N, C, num_patches, _ = unfolded.shape
            act_2d = unfolded.permute(0, 2, 1, 3).reshape(N * num_patches, C * kW)
            return act_2d

        else:
            # Fallback: simple flatten
            return act.reshape(act.size(0), -1).float()

    @contextmanager
    def recording(self):
        """Context manager to enable activation recording."""
        prev_state = self.record_activations
        self.record_activations = True
        try:
            yield self
        finally:
            self.record_activations = prev_state

    def get_activations(self) -> dict[nn.Parameter, torch.Tensor]:
        """Return the recorded activations dict (raw or Gram matrices)."""
        return self._activations

    def clear(self):
        """Free all stored activations from memory."""
        self._activations.clear()

    def remove_hooks(self):
        """Remove all registered hooks from the model."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()
        self.clear()
