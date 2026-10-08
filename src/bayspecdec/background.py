"""
background.py — Background functions for the JAX backend, ``background_fn(x, params) -> (n,)``.

A background is the non-peak part of the spectrum. Unlike a basis it has no ``K`` components
and no amplitude: ``params`` is a flat vector of length ``m`` whose entries get one prior each
(``background_priors`` in ``make_spectral_model``). The model is then

    f(x) = sum_k a_k * basis_fn(x, ...)[k] + background_fn(x, params)

The factories below attach ``n_params`` to the function they return, which
``make_spectral_model`` uses to check ``len(background_priors)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def _with_n_params(fn, n_params: int):
    fn.n_params = n_params
    return fn


def arctan_step_background(x: Array, params: Array) -> Array:
    """
    Absorption edge plus white line of the XANES model (Kashiwamura et al., eq. 2)::

        H (1/2 + arctan((x - E0) / (Gamma/2)) / pi) + A exp(-4 ln2 ((x - E0 - dE) / omega)^2)

    ``params = (H, E0, Gamma, A, dE, omega)``.
    """
    H, E0, gamma, A, dE, omega = params
    step = H * (0.5 + jnp.arctan((x - E0) / (0.5 * gamma)) / jnp.pi)
    white_line = A * jnp.exp(-4.0 * jnp.log(2.0) * ((x - (E0 + dE)) / omega) ** 2)
    return step + white_line


arctan_step_background.n_params = 6  # type: ignore[attr-defined]


def constant_background(x: Array, params: Array) -> Array:
    """Flat offset; ``params = (c,)``."""
    return jnp.full_like(x, params[0])


constant_background.n_params = 1  # type: ignore[attr-defined]


def polynomial_background(degree: int):
    """``sum_j c_j x^j`` for ``j = 0..degree``; ``params = (c_0, ..., c_degree)``."""
    if degree < 0:
        raise ValueError("degree must be >= 0")

    def background(x: Array, params: Array) -> Array:
        return jnp.polyval(params[::-1], x)

    return _with_n_params(background, degree + 1)
