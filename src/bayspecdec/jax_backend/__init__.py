"""
Optional JAX backend: autodiff HMC with batched parallel tempering.

Not imported by ``bayspecdec`` itself, so JAX stays an optional dependency::

    from bayspecdec.jax_backend import make_spectral_model, select_model_size
"""

from .model import (
    BoxTransform,
    JaxModel,
    gaussian_log_likelihood,
    make_spectral_model,
)
from .priors import GammaPrior, JaxPrior, UniformPrior
from .selection import JaxModelRun, select_model_size
from .tempering import JaxParallelTempering, PTConfig

__all__ = [
    "BoxTransform",
    "JaxModel",
    "GammaPrior",
    "JaxPrior",
    "UniformPrior",
    "gaussian_log_likelihood",
    "make_spectral_model",
    "JaxModelRun",
    "select_model_size",
    "JaxParallelTempering",
    "PTConfig",
]
