"""
priors.py — Element-wise priors for the JAX backend.

A prior here is a small immutable object with

* ``lower`` / ``upper``: the support. ``make_spectral_model`` uses it to choose the
  unconstrained reparameterisation (logistic for a finite box, ``exp`` for ``(0, inf)``).
* ``log_prob(x)``: element-wise log density, ``-inf`` outside the support.
* ``sample(key, shape)``: draws used to initialise the chains.

All methods are pure JAX, so they can be jitted and differentiated.
"""

from __future__ import annotations

import math
from typing import Optional, Protocol, Sequence, Union, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

Array = jax.Array


def _log_expm1(x: Array) -> Array:
    # Stable log(exp(x) - 1) for x > 0
    return jnp.where(x <= 1.0, jnp.log(jnp.expm1(x)), x + jnp.log1p(-jnp.exp(-x)))


@runtime_checkable
class JaxPrior(Protocol):
    lower: float
    upper: float

    def log_prob(self, x: Array) -> Array: ...

    def sample(self, key: Array, shape: tuple[int, ...]) -> Array: ...


PriorSpec = Union[JaxPrior, tuple[float, float]]
"""A prior object, or a ``(lower, upper)`` tuple meaning uniform."""


class UniformPrior:
    """Uniform density on ``[lower, upper]``."""

    def __init__(self, lower: float, upper: float):
        if not upper > lower:
            raise ValueError("upper must be greater than lower")
        self.lower = float(lower)
        self.upper = float(upper)
        self._log_density = -float(np.log(self.upper - self.lower))

    def log_prob(self, x: Array) -> Array:
        inside = (x >= self.lower) & (x <= self.upper)
        return jnp.where(inside, self._log_density, -jnp.inf)

    def sample(self, key: Array, shape: tuple[int, ...]) -> Array:
        return jax.random.uniform(key, shape, minval=self.lower, maxval=self.upper)


class GammaPrior:
    """
    Gamma density on ``(0, inf)``.

    Give exactly one of ``rate`` (density ``∝ x^(shape-1) exp(-rate x)``) or
    ``scale = 1 / rate``. The two spellings are explicit because the numpy
    ``GammaPrior(shape, rate)`` samples and exponentiates with its second argument as
    a *scale* but normalises it as a *rate*; ``GammaPrior(shape, scale=s)`` here
    matches the numpy sampling behaviour with the correct normalisation.
    """

    lower = 0.0
    upper = float("inf")

    def __init__(
        self, shape: float, rate: Optional[float] = None, scale: Optional[float] = None
    ):
        if (rate is None) == (scale is None):
            raise ValueError("Pass exactly one of rate or scale")
        if shape <= 0.0:
            raise ValueError("shape must be positive")
        self.shape = float(shape)
        if rate is not None:
            self.rate = float(rate)
        else:
            assert scale is not None  # exactly one of rate / scale was given
            self.rate = 1.0 / float(scale)
        if self.rate <= 0.0:
            raise ValueError("rate/scale must be positive")
        self._log_norm = self.shape * math.log(self.rate) - math.lgamma(self.shape)

    def log_prob(self, x: Array) -> Array:
        positive = x > 0.0
        safe = jnp.where(positive, x, 1.0)  # keeps gradients finite off-support
        out = self._log_norm + (self.shape - 1.0) * jnp.log(safe) - self.rate * safe
        return jnp.where(positive, out, -jnp.inf)

    def sample(self, key: Array, shape: Sequence[int]) -> Array:
        return jax.random.gamma(key, self.shape, shape) / self.rate


class NormalPrior:
    """
    Normal density with mean ``mean`` and standard deviation ``std``, support the real line.

    The numpy ``NormalPrior`` samples with ``std`` correctly but its ``log_prob`` treats
    ``std`` as a precision (and drops a factor); this one is the proper normalised density,
    checked against ``scipy.stats.norm``.
    """

    lower = -float("inf")
    upper = float("inf")

    def __init__(self, mean: float, std: float):
        if std <= 0.0:
            raise ValueError("std must be positive")
        self.mean = float(mean)
        self.std = float(std)
        self._log_norm = -math.log(self.std) - 0.5 * math.log(2.0 * math.pi)

    def log_prob(self, x: Array) -> Array:
        return self._log_norm - 0.5 * ((x - self.mean) / self.std) ** 2

    def sample(self, key: Array, shape: Sequence[int]) -> Array:
        return self.mean + self.std * jax.random.normal(key, shape)


class FermiDiracPrior:
    lower = 0.0
    upper = float("inf")

    def __init__(self, mu: float, temperature: float):
        self.mu = float(mu)
        self.temperature = float(temperature)
        self.log_L = jnp.log(jnp.logaddexp(0.0, self.mu / self.temperature))

    def sample(self, key: Array, shape: Sequence[int]) -> Array:
        L = jnp.exp(self.log_L)
        u = 1.0 - jax.random.uniform(key, shape=shape)
        return self.mu - self.temperature * _log_expm1(u * L)

    def log_prob(self, x: Array) -> Array:
        z = (x - self.mu) / self.temperature
        out = -jnp.log(self.temperature) - self.log_L - jnp.logaddexp(0.0, z)
        return jnp.where(x < 0, -np.inf, out)


def paper_synthetic_priors() -> tuple[JaxPrior, list[JaxPrior]]:
    """
    ``(amplitude_prior, basis_priors)`` of the paper's synthetic setting (Section 3.1),
    for ``make_spectral_model`` with a ``(mu, b)`` basis: Gamma amplitudes, Normal centres,
    Gamma bandwidths. Matches the numpy ``paper_synthetic_prior`` (rates 5 and 0.04).
    """
    return GammaPrior(5.0, rate=5.0), [
        NormalPrior(1.5, 5.0),
        GammaPrior(5.0, rate=0.04),
    ]


def log_scale_prior(lower: float, upper: float) -> UniformPrior:
    """
    Prior for a sampled noise variance ``sigma^2`` in ``[lower, upper]``, expressed on
    ``s = log sigma^2``: uniform in ``s`` is the log-uniform (Jeffreys-like) prior on ``sigma^2``.
    """
    if not (0.0 < lower < upper):
        raise ValueError("need 0 < lower < upper")
    return UniformPrior(math.log(lower), math.log(upper))


def noise_prior_from_variance(sigma2: float, log_std: float = 0.3) -> NormalPrior:
    """
    Prior for a sampled noise variance centred on a known estimate ``sigma2``.

    The sampled noise parameter is ``s = log sigma^2``, so this is a Normal on ``s`` with mean
    ``log(sigma2)`` and standard deviation ``log_std`` (the relative uncertainty of the variance:
    0.3 means roughly a factor ``e**0.3 ~ 1.35`` either way). Use it as ``noise_prior=``.

    ``sigma2`` is a *variance in the units of the spectrum you pass to the model*, so divide a variance
    measured on raw data by the square of whatever you normalised the spectrum by.
    """
    if not (sigma2 > 0.0 and math.isfinite(sigma2)):
        raise ValueError("sigma2 must be positive and finite")
    if not log_std > 0.0:
        raise ValueError("log_std must be positive")
    return NormalPrior(math.log(sigma2), log_std)


def noise_prior_from_dark_frame(
    dark: npt.ArrayLike, scale: float = 1.0, log_std: Optional[float] = None
) -> NormalPrior:
    """
    ``noise_prior=`` centred on the noise variance measured from a dark frame.

    The *variance* of the dark frame sets the noise level; its mean is an offset, which you should
    subtract from the spectrum or model with ``constant_background``.

    ``scale``: the factor you divided the spectrum by when normalising it, so that the variance is
    expressed in the model's units (``var(dark) / scale**2``).

    ``log_std``: width of the Normal on ``log sigma^2``. By default ``max(sqrt(2 / (n - 1)), 0.1)``:
    the first term is the sampling uncertainty of a variance estimated from ``n`` independent Gaussian
    pixels; the floor keeps a very large dark frame from pinning ``sigma^2`` when the spectrum has noise
    the dark frame does not (shot noise, correlated pixels, ...). Pass your own value to override.
    """
    values = np.asarray(dark, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("need at least two dark-frame values")
    if not scale > 0.0:
        raise ValueError("scale must be positive")
    variance = float(np.var(values, ddof=1)) / scale**2
    if not variance > 0.0:
        raise ValueError("the dark frame has zero variance")
    if log_std is None:
        log_std = max(math.sqrt(2.0 / (values.size - 1)), 0.1)
    return noise_prior_from_variance(variance, log_std)


def as_prior(spec: PriorSpec) -> JaxPrior:
    """Accept a prior object or a ``(lower, upper)`` tuple (shorthand for uniform)."""
    if isinstance(spec, JaxPrior):
        return spec
    lower, upper = spec
    return UniformPrior(lower, upper)
