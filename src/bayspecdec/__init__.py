"""
Bayesian spectral deconvolution with autodiff HMC/NUTS and batched exchange Monte Carlo (JAX).

Importing the package enables float64 in JAX (sigma^2 ~ 1e-4 likelihoods are not safe in float32).
"""

from .model import BoxTransform, JaxModel, SpectralJaxModel, make_spectral_model
from .basis import gaussian_basis, lorentzian_basis
from .background import (
    BackgroundFunction,
    arctan_step_background,
    constant_background,
    polynomial_background,
    with_n_params,
)
from .likelihoods import (
    gaussian_log_likelihood,
    gaussian_noise_likelihood,
    heteroscedastic_gaussian_log_likelihood,
    poisson_gaussian_log_likelihood,
    poisson_log_likelihood,
)
from .priors import (
    FermiDiracPrior,
    GammaPrior,
    JaxPrior,
    NormalPrior,
    PriorSpec,
    UniformPrior,
    log_scale_prior,
    paper_synthetic_priors,
)
from .tempering import ExchangeResult, JaxParallelTempering, PTConfig
from .selection import JaxModelRun, select_model_size
from .evidence import EvidenceEstimate, estimate_evidence
from .functions import beta_schedule
from .data import make_paper_like_synthetic_data

__all__ = [
    "BoxTransform",
    "JaxModel",
    "SpectralJaxModel",
    "make_spectral_model",
    "gaussian_basis",
    "lorentzian_basis",
    "BackgroundFunction",
    "with_n_params",
    "arctan_step_background",
    "constant_background",
    "polynomial_background",
    "gaussian_log_likelihood",
    "gaussian_noise_likelihood",
    "heteroscedastic_gaussian_log_likelihood",
    "poisson_gaussian_log_likelihood",
    "poisson_log_likelihood",
    "FermiDiracPrior",
    "GammaPrior",
    "JaxPrior",
    "NormalPrior",
    "PriorSpec",
    "UniformPrior",
    "log_scale_prior",
    "paper_synthetic_priors",
    "ExchangeResult",
    "JaxParallelTempering",
    "PTConfig",
    "JaxModelRun",
    "select_model_size",
    "EvidenceEstimate",
    "estimate_evidence",
    "beta_schedule",
    "make_paper_like_synthetic_data",
]
