"""
Bayesian spectral deconvolution with exchange Monte Carlo (parallel tempering).
Refactored into a modular, extensible package.
"""

from .basis import GaussianBasis, LorentzianBasis
from .parameters import DefaultParameterization
from .priors import (
    GammaPrior,
    NormalPrior,
    IndependentProductPrior,
    paper_synthetic_prior,
)
from .likelihoods import GaussianNoise
from .models import SpectralModel
from .samplers.metropolis import RandomWalkMetropolis, metropolis_kernel_factory
from .tempering import ParallelTempering, ExchangeResult
from .evidence import EvidenceEstimate, estimate_evidence
from .model_selection import ModelRun, select_model_size
from .functions import beta_schedule
from .data import make_paper_like_synthetic_data

__all__ = [
    "GaussianBasis",
    "LorentzianBasis",
    "DefaultParameterization",
    "GammaPrior",
    "NormalPrior",
    "IndependentProductPrior",
    "paper_synthetic_prior",
    "GaussianNoise",
    "SpectralModel",
    "RandomWalkMetropolis",
    "metropolis_kernel_factory",
    "ParallelTempering",
    "ExchangeResult",
    "EvidenceEstimate",
    "estimate_evidence",
    "ModelRun",
    "select_model_size",
    "beta_schedule",
    "make_paper_like_synthetic_data",
]
