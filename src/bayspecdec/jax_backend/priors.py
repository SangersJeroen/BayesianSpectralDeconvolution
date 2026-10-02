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
from typing import Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


@runtime_checkable
class JaxPrior(Protocol):
    lower: float
    upper: float

    def log_prob(self, x: Array) -> Array: ...

    def sample(self, key: Array, shape: tuple[int, ...]) -> Array: ...


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
        self.rate = float(rate) if rate is not None else 1.0 / float(scale)
        if self.rate <= 0.0:
            raise ValueError("rate/scale must be positive")
        self._log_norm = self.shape * math.log(self.rate) - math.lgamma(self.shape)

    def log_prob(self, x: Array) -> Array:
        positive = x > 0.0
        safe = jnp.where(positive, x, 1.0)  # keeps gradients finite off-support
        out = self._log_norm + (self.shape - 1.0) * jnp.log(safe) - self.rate * safe
        return jnp.where(positive, out, -jnp.inf)

    def sample(self, key: Array, shape: tuple[int, ...]) -> Array:
        return jax.random.gamma(key, self.shape, shape) / self.rate


def as_prior(spec) -> JaxPrior:
    """Accept a prior object or a ``(lower, upper)`` tuple (shorthand for uniform)."""
    if isinstance(spec, JaxPrior):
        return spec
    lower, upper = spec
    return UniformPrior(lower, upper)
