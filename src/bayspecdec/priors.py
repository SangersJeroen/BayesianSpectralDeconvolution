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
    def __init__(self, mean: float, precision: float):
        self.mean = float(mean)
        self.precision = float(precision)

    def sample(self, rng: np.random.Generator, size: int = 1):
        return rng.normal(loc=self.mean, scale=1.0 / np.sqrt(self.precision), size=size)

    def log_prob(self, x: Array) -> Array:
        return (
            0.5 * np.log(self.precision / (2.0 * np.pi))
            - 0.5 * self.precision * (x - self.mean) ** 2
        )


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
        centers=NormalPrior(mean=1.5, precision=5.0),
        bandwidths=GammaPrior(shape=5.0, rate=0.04),
    )
