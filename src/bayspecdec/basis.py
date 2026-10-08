"""basis.py — JAX versions of the numpy basis functions, ``basis_fn(x, params) -> (K, n)``."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def gaussian_basis(x: Array, params: Array) -> Array:
    """``exp(-b (x - mu)^2 / 2)``; ``params = (mu, b)`` of shape ``(2, K)``."""
    mu, b = params
    return jnp.exp(-0.5 * b[:, None] * (x[None, :] - mu[:, None]) ** 2)


def lorentzian_basis(x: Array, params: Array) -> Array:
    """``1 / (1 + ((x - mu) / gamma)^2)``; ``params = (mu, gamma)`` of shape ``(2, K)``."""
    mu, gamma = params
    return 1.0 / (1.0 + ((x[None, :] - mu[:, None]) / gamma[:, None]) ** 2)
