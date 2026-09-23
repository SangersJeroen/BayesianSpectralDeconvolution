import numba
from numba.experimental.jitclass.decorators import jitclass
import numpy as np
from typing import Protocol, runtime_checkable
from .parameters import Parameterization

import math

Array = np.ndarray


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
        return rng.gamma(shape=self.shape, scale=1.0 / self.rate, size=size)

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
