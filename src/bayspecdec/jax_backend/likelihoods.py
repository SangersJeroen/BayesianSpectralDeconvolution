"""
likelihoods.py — Observation models ``(y, prediction) -> scalar`` for the JAX backend.

All are pure JAX, fully normalised and differentiable in ``prediction``, so they plug
into ``make_spectral_model(likelihood=...)``.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

Array = jax.Array
_TINY = 1e-12


def gaussian_log_likelihood(sigma2: float) -> Callable[[Array, Array], Array]:
    """Fully normalised Gaussian log-likelihood ``(y, prediction) -> scalar``."""

    def log_prob(y: Array, prediction: Array) -> Array:
        residual = y - prediction
        return -0.5 * residual.size * jnp.log(2.0 * jnp.pi * sigma2) - 0.5 * jnp.sum(
            residual**2
        ) / sigma2

    return log_prob


def poisson_log_likelihood() -> Callable[[Array, Array], Array]:
    """
    ``y_i ~ Poisson(prediction_i)``: ``sum_i y_i log(lam_i) - lam_i - log(y_i!)``.

    ``prediction`` is clamped to a tiny positive floor so the value and gradient stay
    finite when the model dips to or below zero (the likelihood there is ~ -inf anyway).
    """

    def log_prob(y: Array, prediction: Array) -> Array:
        lam = jnp.maximum(prediction, _TINY)
        return jnp.sum(y * jnp.log(lam) - lam - gammaln(y + 1.0))

    return log_prob


def poisson_gaussian_log_likelihood(
    sigma2: float, half_width: int = 12
) -> Callable[[Array, Array], Array]:
    """
    Poisson counts read out with Gaussian noise::

        k_i ~ Poisson(lam_i),  y_i | k_i ~ N(k_i, sigma2)
        p(y_i) = sum_k Poisson(k | lam_i) N(y_i | k, sigma2)

    The numpy class sums until a tolerance is met, which needs data-dependent loops. Here the
    sum runs over a *fixed* window of ``2 * half_width + 1`` integers centred on the
    summand's mode, so it is jittable and differentiable. The summand is approximately
    Gaussian in ``k`` with mean ``lam (sigma2 + y) / (sigma2 + lam)`` and standard deviation
    ``s = sqrt(lam sigma2 / (lam + sigma2)) <= min(sqrt(lam), sqrt(sigma2))``, so the window
    covers ``+-half_width / s`` standard deviations; the default is exact to machine
    precision whenever ``s <= 1``. Raise ``half_width`` for ``sigma2`` and ``lam`` both
    well above 1. The window centre is held constant under ``grad`` (it only selects terms).
    """
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be strictly positive")
    if half_width < 1:
        raise ValueError("half_width must be >= 1")
    log_pref = -0.5 * math.log(2.0 * math.pi * sigma2)
    offsets = jnp.arange(-half_width, half_width + 1, dtype=float)

    def log_prob(y: Array, prediction: Array) -> Array:
        lam = jnp.maximum(prediction, _TINY)
        centre = jax.lax.stop_gradient(
            jnp.maximum(jnp.round(lam * (sigma2 + y) / (sigma2 + lam)), 0.0)
        )
        # Keep the whole window non-negative: shift it up if it would cross k = 0.
        start = jnp.maximum(centre - half_width, 0.0)
        k = start[..., None] + (offsets + half_width)  # (..., 2h+1)
        yb, lb = y[..., None], lam[..., None]
        log_a = k * jnp.log(lb) - lb - gammaln(k + 1.0) + log_pref - (yb - k) ** 2 / (2.0 * sigma2)
        return jnp.sum(jax.scipy.special.logsumexp(log_a, axis=-1))

    return log_prob
