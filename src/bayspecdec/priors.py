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
        out = (
            self.shape * np.log(self.rate)
            - math.lgamma(self.shape)
            + (self.shape - 1.0) * np.log(x)
            - self.rate * x
        )
        return np.where(x <= 0, -np.inf, out)


@jitclass([("mean", numba.float64), ("precision", numba.float64)])
class NormalPrior:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def sample(self, rng: np.random.Generator, size: int = 1):
        return rng.normal(loc=self.mean, scale=self.std, size=size)

    def log_prob(self, x: Array) -> Array:
        return (
            0.5 * np.log(self.std / (2.0 * np.pi))
            - 0.5 * self.std * (x - self.mean) ** 2
        )


@jitclass([("lower", numba.float64), ("upper", numba.float64)])
class UniformPrior:
    def __init__(self, lower: float, upper: float):
        self.lower = float(lower)
        self.upper = float(upper)

    def sample(self, rng: np.random.Generator, size: int = 1):
        return rng.uniform(low=self.lower, high=self.upper, size=size)

    def log_prob(self, x: Array) -> Array:
        return np.broadcast_to(np.log(1 / abs(self.upper - self.lower)), shape=x.shape)


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
        out = -np.log(self.temperature) - np.exp(self.log_L) - np.logaddexp(0.0, z)
        return np.where(x < 0, -np.inf, out)

class IndependentProductPrior:
    """
    Combines independent priors for a, mu, b (the paper's parameters).
    This assumes a specific parameterization structure: (a, mu, b)
    """

    def __init__(self, amplitudes, centers, bandwidths):
        self.amplitudes = amplitudes
        self.centers = centers
        self.bandwidths = bandwidths

    def sample(
        self, rng: np.random.Generator, parameterization: Parameterization
    ) -> Array:
        K = parameterization.K
        a = self.amplitudes.sample(rng, K)
        mu = self.centers.sample(rng, K)
        b = self.bandwidths.sample(rng, K)
        return parameterization.pack((a, mu, b))

    def log_prob(self, theta: Array, parameterization: Parameterization) -> float:
        a, mu, b = parameterization.unpack(theta)

        lp_a = np.sum(self.amplitudes.log_prob(a))
        lp_mu = np.sum(self.centers.log_prob(mu))
        lp_b = np.sum(self.bandwidths.log_prob(b))

        return float(lp_a + lp_mu + lp_b)


def paper_synthetic_prior() -> Prior:
    """Returns the synthetic prior from the original paper (Section 3.1)."""
    return IndependentProductPrior(
        amplitudes=GammaPrior(shape=5.0, rate=5.0),
        centers=NormalPrior(mean=1.5, std=5.0),
        bandwidths=GammaPrior(shape=5.0, rate=0.04),
    )
