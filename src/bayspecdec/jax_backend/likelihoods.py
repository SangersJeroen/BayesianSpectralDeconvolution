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
    sigma2: float, half_width: int = 12, gain: float = 1.0
) -> Callable[[Array, Array], Array]:
    """
    Poisson counts read out with Gaussian noise::

        k_i ~ Poisson(lam_i),  y_i | k_i ~ N(gain * k_i, sigma2)
        p(y_i) = sum_k Poisson(k | lam_i) N(y_i | gain * k, sigma2)

    ``prediction`` is the mean of ``y`` (``gain * lam``); ``sigma2`` is the readout variance in
    the units of ``y``. ``gain`` is the detector gain (counts per detected event); with
    ``gain=1`` the data are plain event counts. Then ``var(y) = gain * mean(y) + sigma2``.

    Call it to get the likelihood: ``make_spectral_model(..., likelihood=poisson_gaussian_log_likelihood(sigma2))``.

    The numpy class sums until a tolerance is met, which needs data-dependent loops. Here the
    sum runs over a *fixed* window of ``2 * half_width + 1`` integers centred on the
    summand's mode, so it is jittable and differentiable. In event units the summand is
    approximately Gaussian in ``k`` with mean ``lam (s2 + y) / (s2 + lam)`` and standard
    deviation ``s = sqrt(lam s2 / (lam + s2)) <= min(sqrt(lam), sqrt(s2))`` with
    ``s2 = sigma2 / gain**2``, so the window covers ``+-half_width / s`` standard deviations.
    Choose ``half_width >= 8 * sqrt(min(lam_max, s2))``; the default is exact to machine
    precision when ``s <= 1`` and is far too narrow for a large readout noise, where
    ``heteroscedastic_gaussian_log_likelihood`` is the cheaper (and, for large counts,
    equally accurate) choice. The window centre is held constant under ``grad``.
    """
    if jnp.ndim(sigma2) != 0 or jnp.ndim(gain) != 0:
        raise TypeError(
            "sigma2 and gain must be scalars. This function returns the likelihood: pass "
            "likelihood=poisson_gaussian_log_likelihood(sigma2), not the function itself"
        )
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be strictly positive")
    if gain <= 0.0:
        raise ValueError("gain must be strictly positive")
    if half_width < 1:
        raise ValueError("half_width must be >= 1")
    s2 = float(sigma2) / float(gain) ** 2
    log_gain = math.log(float(gain))
    log_pref = -0.5 * math.log(2.0 * math.pi * s2)
    offsets = jnp.arange(-half_width, half_width + 1, dtype=float)

    def log_prob(y: Array, prediction: Array) -> Array:
        y = y / gain  # event units
        lam = jnp.maximum(prediction / gain, _TINY)
        centre = jax.lax.stop_gradient(jnp.maximum(jnp.round(lam * (s2 + y) / (s2 + lam)), 0.0))
        # Keep the whole window non-negative: shift it up if it would cross k = 0.
        start = jnp.maximum(centre - half_width, 0.0)
        k = start[..., None] + (offsets + half_width)  # (..., 2h+1)
        yb, lb = y[..., None], lam[..., None]
        log_a = k * jnp.log(lb) - lb - gammaln(k + 1.0) + log_pref - (yb - k) ** 2 / (2.0 * s2)
        # Density of y = density of y / gain, divided by gain.
        return jnp.sum(jax.scipy.special.logsumexp(log_a, axis=-1)) - y.size * log_gain

    return log_prob


def heteroscedastic_gaussian_log_likelihood(
    read_var: float, gain: float = 1.0
) -> Callable[[Array, Array], Array]:
    """
    Gaussian approximation of Poisson + readout noise: ``y_i ~ N(pred_i, gain * pred_i + read_var)``.

    This is the large-count limit of ``poisson_gaussian_log_likelihood(read_var, gain=gain)``
    (it matches the mean and variance), costs one term per bin instead of ``2 * half_width + 1``,
    and is the usual choice for detector data with hundreds of counts or more. The variance is
    part of the density, so the gradient includes its dependence on the prediction.
    """
    if read_var <= 0.0 or gain <= 0.0:
        raise ValueError("read_var and gain must be strictly positive")

    def log_prob(y: Array, prediction: Array) -> Array:
        var = gain * jnp.maximum(prediction, 0.0) + read_var
        return -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * var) + (y - prediction) ** 2 / var)

    return log_prob
