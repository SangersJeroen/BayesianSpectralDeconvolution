"""
Optional JAX backend: autodiff HMC with batched parallel tempering.

Not imported by ``bayspecdec`` itself, so JAX stays an optional dependency::

    from bayspecdec.jax_backend import make_spectral_model, select_model_size
"""

from .model import (
    BoxTransform,
    JaxModel,
    make_spectral_model,
)
from .basis import gaussian_basis, lorentzian_basis
from .background import (
    arctan_step_background,
    constant_background,
    polynomial_background,
)
from .likelihoods import (
    gaussian_log_likelihood,
    gaussian_noise_likelihood,
    heteroscedastic_gaussian_log_likelihood,
    poisson_gaussian_log_likelihood,
    poisson_log_likelihood,
)
from .priors import (
    GammaPrior,
    JaxPrior,
    NormalPrior,
    UniformPrior,
    FermiDiracPrior,
    log_scale_prior,
    paper_synthetic_priors,
)
from .selection import JaxModelRun, select_model_size
from .tempering import JaxParallelTempering, PTConfig

__all__ = [
    "BoxTransform",
    "JaxModel",
    "GammaPrior",
    "JaxPrior",
    "NormalPrior",
    "UniformPrior",
    "FermiDiracPrior",
    "paper_synthetic_priors",
    "log_scale_prior",
    "gaussian_noise_likelihood",
    "gaussian_basis",
    "lorentzian_basis",
    "arctan_step_background",
    "constant_background",
    "polynomial_background",
    "heteroscedastic_gaussian_log_likelihood",
    "poisson_log_likelihood",
    "poisson_gaussian_log_likelihood",
    "gaussian_log_likelihood",
    "make_spectral_model",
    "JaxModelRun",
    "select_model_size",
    "JaxParallelTempering",
    "PTConfig",
]
