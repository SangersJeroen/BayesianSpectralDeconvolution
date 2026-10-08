"""
background.py — Background functions for the JAX backend, ``background_fn(x, params) -> (n,)``.

A background is the non-peak part of the spectrum. Unlike a basis it has no ``K`` components
and no per-component amplitude: ``params`` is a flat vector of length ``m`` whose entries get one prior each
(``background_priors`` in ``make_spectral_model``). The model is then

    f(x) = sum_k a_k * basis_fn(x, ...)[k] + background_fn(x, params)

Each function below carries an ``n_params`` attribute, which ``make_spectral_model`` uses to check
``len(background_priors)``.
"""

from __future__ import annotations

from typing import Callable, Protocol, cast

import jax
import jax.numpy as jnp

Array = jax.Array


class BackgroundFunction(Protocol):
    """``(x, params) -> (n,)`` with the number of parameters it expects as ``n_params``."""

    n_params: int

    def __call__(self, x: Array, params: Array) -> Array: ...


def with_n_params(
    fn: Callable[[Array, Array], Array], n_params: int
) -> BackgroundFunction:
    """Attach ``n_params`` to a plain ``(x, params) -> (n,)`` function (use it for custom backgrounds)."""
    setattr(fn, "n_params", n_params)
    return cast(BackgroundFunction, fn)


def _arctan_step(x: Array, params: Array) -> Array:
    H, E0, gamma, A, dE, omega = params
    step = H * (0.5 + jnp.arctan((x - E0) / (0.5 * gamma)) / jnp.pi)
    white_line = A * jnp.exp(-4.0 * jnp.log(2.0) * ((x - (E0 + dE)) / omega) ** 2)
    result: Array = step + white_line
    return result


arctan_step_background = with_n_params(_arctan_step, 6)
arctan_step_background.__doc__ = """
Absorption edge plus white line of the XANES model (Kashiwamura et al., eq. 2)::

    H (1/2 + arctan((x - E0) / (Gamma/2)) / pi) + A exp(-4 ln2 ((x - E0 - dE) / omega)^2)

``params = (H, E0, Gamma, A, dE, omega)``.
"""


def _constant(x: Array, params: Array) -> Array:
    return jnp.full_like(x, params[0])


constant_background = with_n_params(_constant, 1)
constant_background.__doc__ = "Flat offset; ``params = (c,)``."


def polynomial_background(degree: int) -> BackgroundFunction:
    """``sum_j c_j x^j`` for ``j = 0..degree``; ``params = (c_0, ..., c_degree)``."""
    if degree < 0:
        raise ValueError("degree must be >= 0")

    def background(x: Array, params: Array) -> Array:
        result: Array = jnp.polyval(params[::-1], x)
        return result

    return with_n_params(background, degree + 1)
