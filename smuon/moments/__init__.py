"""
Second-order moment tracking for adaptive preconditioning.

This module provides different strategies for tracking second-order moment statistics
(typically gradient variance) used for adaptive preconditioning in optimization.

Available Moment Types
----------------------
- **adam**: Adam-style exponential moving average with fixed exponent (-1/2)
- **adagrad**: Adagrad-style accumulated second moment with fixed exponent (-1/2)
- **adafactor**: Adafactor-style factorized second moment with fixed exponent (-1/4)
- **sania**: SANIA optimizer-style moment with fixed exponent (-1/2)
- **sgd**: Plain SGD (no second moment tracking)

- **padam**: Parametric Adam (exponent scales with p*)
- **padagrad**: Parametric Adagrad (exponent scales with p*)
- **padafactor**: Parametric Adafactor (exponent scales with p*)
- **psania**: Parametric SANIA (exponent scales with p*)

The "parametric" variants (prefixed with 'p') dynamically adjust the preconditioner
exponent based on the current p* value from the Schatten-p norm optimization.

Usage Example
-------------
```python
from smuon.moments import create_moment

# Create an Adam-style moment tracker
moment = create_moment(
    moment_type="adam",
    beta2=0.999,
    eps=1e-8,
    use_bias_correction=True
)

# Update the moment and get preconditioning matrix
state = {}  # Optimizer state dictionary
gradient = ...  # Current gradient tensor
D_t = moment.update(state, gradient)

# D_t contains the element-wise preconditioning factors
preconditioned_grad = D_t * gradient
```

Parametric Moments
------------------
For parametric variants, specify the current p value:

```python
moment = create_moment(
    moment_type="padam",
    p=2.5,  # Current Schatten-p norm exponent
    beta2=0.999,
    eps=1e-8,
    use_bias_correction=True
)

# Update p dynamically during training
moment.p = new_p_value
```

Factory Function
----------------
Use `create_moment()` to instantiate moments, or use `is_parametric()` to check
whether a moment type requires dynamic p values.
"""

from smuon.moments.base import SecondOrderMoment
from smuon.moments.adafactor import AdafactorMoment
from smuon.moments.adam import AdamMoment
from smuon.moments.adagrad import AdaGradMoment
from smuon.moments.sania import SaniaMoment
from smuon.moments.registry import (
    MOMENT_CLASSES,
    MOMENT_REGISTRY,
    create_moment,
    is_parametric,
)

__all__ = [
    "SecondOrderMoment",
    "AdamMoment",
    "AdaGradMoment",
    "AdafactorMoment",
    "SaniaMoment",
    "MOMENT_CLASSES",
    "MOMENT_REGISTRY",
    "create_moment",
    "is_parametric",
]
