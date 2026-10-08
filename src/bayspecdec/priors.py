import numba
from numba.experimental.jitclass.decorators import jitclass
import numpy as np
from typing import Protocol, runtime_checkable
from .parameters import Parameterization

import math

Array = np.ndarray

@numba.njit
def _log_expm1(x):
    # Stable log(exp(x) - 1) for x > 0
    return np.where(x <= 1.0, np.log(np.expm1(x)), x + np.log1p(-np.exp(-x)))

@runtime_checkable
class Prior(Protocol):
    def sample(
        self, rng: np.random.Generator, parameterization: Parameterization
    ) -> Array: ...

    def log_prob(self, theta: Array, parameterization: Parameterization) -> float: ...


@jitclass([("shape", numba.float64), ("rate", numba.float64)])
class GammaPrior:
    def __init__(self, shape: float, rate: float):
        self.shape = float(shape)
        self.rate = float(rate)

    def sample(self, rng: np.random.Generator, size: int = 1):
        return rng.gamma(shape=self.shape, scale=1/self.rate, size=size)

    def log_prob(self, x: Array) -> Array:
        safe = np.where(x > 0, x, 1.0)
        out = (
            self.shape * np.log(self.rate)
            - math.lgamma(self.shape)
            + (self.shape - 1.0) * np.log(safe)
            - self.rate * safe
        )
        return np.where(x <= 0, -np.inf, out)


@jitclass([("mean", numba.float64), ("std", numba.float64)])
class NormalPrior:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def sample(self, rng: np.random.Generator, size: int = 1):
        return rng.normal(loc=self.mean, scale=self.std, size=size)

    def log_prob(self, x: Array) -> Array:
        return (
            -np.log(self.std)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * ((x - self.mean) / self.std) ** 2
        )


@jitclass([("lower", numba.float64), ("upper", numba.float64)])
class UniformPrior:
    def __init__(self, lower: float, upper: float):
        self.lower = float(lower)
        self.upper = float(upper)

    def sample(self, rng: np.random.Generator, size: int = 1):
        return rng.uniform(low=self.lower, high=self.upper, size=size)

    def log_prob(self, x: Array) -> Array:
        inside = (x >= self.lower) & (x <= self.upper)
        return np.where(inside, -np.log(self.upper - self.lower), -np.inf)


@jitclass([
    ("mu", numba.float64),
    ("temperature", numba.float64),   # this is kT, same units as mu
    ("log_L", numba.float64),         # log of L = ln(1 + exp(mu/kT))
])
class FermiDiracPrior:
    def __init__(self, mu: float, temperature: float):
        self.mu = float(mu)
        self.temperature = float(temperature)
        # L = log1p(exp(mu/kT)) computed stably via logaddexp(0, mu/kT)
        self.log_L = np.log(np.logaddexp(0.0, self.mu / self.temperature))

    def sample(self, rng: np.random.Generator, size: int = 1):
        L = np.exp(self.log_L)
        u = 1.0 - rng.random(size)          # u in (0, 1]
        return self.mu - self.temperature * _log_expm1(u * L)

    def log_prob(self, x):
        z = (x - self.mu) / self.temperature
        out = -np.log(self.temperature) - self.log_L - np.logaddexp(0.0, z)
        return np.where(x < 0, -np.inf, out)


def prior_support(prior) -> tuple[float, float]:
    """``(lower, upper)`` support of one of the element-wise priors, used to pick the transform."""
    name = type(prior).__name__
    if name == "UniformPrior":
        return float(prior.lower), float(prior.upper)
    if name in ("GammaPrior", "FermiDiracPrior"):
        return 0.0, float("inf")
    if name == "NormalPrior":
        return -float("inf"), float("inf")
    raise TypeError(f"Unknown support for prior {name!r}; give the bounds explicitly")


def log_scale_prior(lower: float, upper: float) -> "UniformPrior":
    """
    Prior for a sampled noise scale ``sigma^2`` in ``[lower, upper]``, expressed on ``s = log sigma^2``:
    a uniform density on ``s`` is the log-uniform (Jeffreys-like) prior on ``sigma^2``.
    """
    if not (0.0 < lower < upper):
        raise ValueError("need 0 < lower < upper")
    return UniformPrior(np.log(lower), np.log(upper))


class ProductPrior:
    """
    Independent element-wise priors for the ``BlockParameterization`` layout: one prior for the
    amplitudes, one per basis-parameter block, and optionally one per noise hyperparameter.
    """

    def __init__(self, amplitude, basis_priors, noise_priors=()):
        self.amplitude = amplitude
        self.basis_priors = list(basis_priors)
        self.noise_priors = list(noise_priors)

    def block_bounds(self) -> list[tuple[float, float]]:
        return [prior_support(p) for p in [self.amplitude, *self.basis_priors]]

    def noise_bounds(self) -> list[tuple[float, float]]:
        return [prior_support(p) for p in self.noise_priors]

    def _block_priors(self):
        return [self.amplitude, *self.basis_priors]

    def sample(self, rng: np.random.Generator, parameterization: Parameterization) -> Array:
        K = parameterization.K
        blocks = [p.sample(rng, K) for p in self._block_priors()]
        blocks += [p.sample(rng, 1) for p in self.noise_priors]
        return parameterization.pack(blocks)

    def log_prob(self, theta: Array, parameterization: Parameterization) -> float:
        total = 0.0
        for p, block in zip(self._block_priors(), parameterization.unpack(theta)):
            total += np.sum(p.log_prob(block))
        if self.noise_priors:
            noise = parameterization.noise(theta)
            for p, s in zip(self.noise_priors, noise):
                total += np.sum(p.log_prob(np.atleast_1d(s)))
        return float(total)


class IndependentProductPrior(ProductPrior):
    """
    Combines independent priors for a, mu, b (the paper's parameters).
    This assumes a specific parameterization structure: (a, mu, b)
    """

    def __init__(self, amplitudes, centers, bandwidths, noise=None):
        super().__init__(
            amplitudes, [centers, bandwidths], [] if noise is None else [noise]
        )
        self.amplitudes = amplitudes
        self.centers = centers
        self.bandwidths = bandwidths


def paper_synthetic_prior() -> Prior:
    """Returns the synthetic prior from the original paper (Section 3.1)."""
    return IndependentProductPrior(
        amplitudes=GammaPrior(shape=5.0, rate=5.0),
        centers=NormalPrior(mean=1.5, std=5.0),
        bandwidths=GammaPrior(shape=5.0, rate=0.04),
    )
