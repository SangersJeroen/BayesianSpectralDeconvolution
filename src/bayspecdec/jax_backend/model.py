"""
model.py — Pure-JAX model definition for the autodiff HMC / parallel-tempering backend.

Everything the sampler touches is a pure function of ``theta`` so that it can be
``jit``-ed, ``vmap``-ed over temperatures and differentiated with ``jax.grad``.
The basis may therefore be any differentiable JAX function, including a neural
network, as long as it has the signature ``basis_fn(x, params) -> (K, n)`` with
``params`` of shape ``(n_basis_params, K)``.

Importing this module enables float64 in JAX (sigma^2 ~ 1e-4 likelihoods are not
safe in float32).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

Array = jax.Array
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Unconstrained reparameterisation
# ---------------------------------------------------------------------------


class BoxTransform:
    """
    Maps unconstrained ``z`` to box-constrained ``theta`` element-wise.

    * both bounds finite  -> logistic map onto (lower, upper)
    * only lower finite   -> ``lower + exp(z)``
    * no finite bound     -> identity

    HMC then never leaves the prior support, and ``log_jac`` supplies
    ``log |d theta / d z|`` which must be added to the log density in z-space.
    """

    def __init__(self, lower: Sequence[float], upper: Sequence[float]):
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(np.isfinite(upper) & ~np.isfinite(lower)):
            raise ValueError("Upper-only bounds are not supported")

        both = np.isfinite(lower) & np.isfinite(upper)
        low_only = np.isfinite(lower) & ~np.isfinite(upper)
        self.ndim = lower.size
        self._both = jnp.asarray(both)
        self._low = jnp.asarray(low_only)
        # Finite placeholders keep the unused ``where`` branches NaN-free.
        self._lo = jnp.asarray(np.where(np.isfinite(lower), lower, 0.0))
        self._width = jnp.asarray(np.where(both, upper - lower, 1.0))
        self.lower = lower
        self.upper = upper

    def to_theta(self, z: Array) -> Array:
        bounded = self._lo + self._width * jax.nn.sigmoid(z)
        positive = self._lo + jnp.exp(z)
        return jnp.where(self._both, bounded, jnp.where(self._low, positive, z))

    def to_z(self, theta: Array) -> Array:
        u = jnp.clip((theta - self._lo) / self._width, _EPS, 1.0 - _EPS)
        bounded = jnp.log(u) - jnp.log1p(-u)
        positive = jnp.log(jnp.maximum(theta - self._lo, _EPS))
        return jnp.where(self._both, bounded, jnp.where(self._low, positive, theta))

    def log_jac(self, z: Array) -> Array:
        bounded = jnp.log(self._width) + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z)
        return jnp.sum(jnp.where(self._both, bounded, jnp.where(self._low, z, 0.0)))


# ---------------------------------------------------------------------------
# Generic model container
# ---------------------------------------------------------------------------


@dataclass
class JaxModel:
    """
    Minimal contract the sampler needs.

    Attributes
    ----------
    log_likelihood:
        ``theta -> scalar`` fully normalised log p(y | theta).
    log_prior:
        ``theta -> scalar`` log p(theta) in theta-space (no Jacobian).
    transform:
        Unconstrained reparameterisation.
    sample_prior:
        ``key -> theta`` draw used to initialise the chains.
    """

    ndim: int
    log_likelihood: Callable[[Array], Array]
    log_prior: Callable[[Array], Array]
    transform: BoxTransform
    sample_prior: Callable[[Array], Array]
    K: Optional[int] = None

    def log_posterior(self, theta: Array) -> Array:
        return self.log_likelihood(theta) + self.log_prior(theta)


# ---------------------------------------------------------------------------
# Spectral-model convenience builder
# ---------------------------------------------------------------------------


def gaussian_log_likelihood(sigma2: float) -> Callable[[Array, Array], Array]:
    """Fully normalised Gaussian log-likelihood ``(y, prediction) -> scalar``."""

    def log_prob(y: Array, prediction: Array) -> Array:
        residual = y - prediction
        return -0.5 * residual.size * jnp.log(2.0 * jnp.pi * sigma2) - 0.5 * jnp.sum(
            residual**2
        ) / sigma2

    return log_prob


def make_spectral_model(
    x: np.ndarray,
    y: np.ndarray,
    basis_fn: Callable[[Array, Array], Array],
    K: int,
    amplitude_bounds: tuple[float, float],
    basis_param_bounds: Sequence[tuple[float, float]],
    sigma2: Optional[float] = None,
    likelihood: Optional[Callable[[Array, Array], Array]] = None,
) -> JaxModel:
    """
    ``y ~ sum_k a_k * basis_fn(x, params)[k]`` with uniform priors on boxes.

    theta layout: ``[a_1..a_K, p1_1..p1_K, p2_1..p2_K, ...]`` with one block of
    K values per entry of ``basis_param_bounds``. ``basis_fn`` receives the
    parameter blocks stacked as ``(n_basis_params, K)`` and returns ``(K, n)``.
    Pass either ``sigma2`` (Gaussian noise) or a custom ``likelihood(y, pred)``.
    """
    if (sigma2 is None) == (likelihood is None):
        raise ValueError("Pass exactly one of sigma2 or likelihood")
    if likelihood is None:
        likelihood = gaussian_log_likelihood(sigma2)

    n_blocks = 1 + len(basis_param_bounds)
    lower = np.repeat([amplitude_bounds[0]] + [b[0] for b in basis_param_bounds], K)
    upper = np.repeat([amplitude_bounds[1]] + [b[1] for b in basis_param_bounds], K)
    transform = BoxTransform(lower, upper)

    xj = jnp.asarray(x, dtype=float)
    yj = jnp.asarray(y, dtype=float)
    log_density = -float(np.sum(np.log(upper - lower)))
    lower_j, upper_j = jnp.asarray(lower), jnp.asarray(upper)

    def predict(theta: Array) -> Array:
        blocks = theta.reshape(n_blocks, K)
        return blocks[0] @ basis_fn(xj, blocks[1:])

    def log_likelihood(theta: Array) -> Array:
        return likelihood(yj, predict(theta))

    def log_prior(theta: Array) -> Array:
        inside = jnp.all((theta >= lower_j) & (theta <= upper_j))
        return jnp.where(inside, log_density, -jnp.inf)

    def sample_prior(key: Array) -> Array:
        return jax.random.uniform(key, (n_blocks * K,), minval=lower_j, maxval=upper_j)

    model = JaxModel(
        ndim=n_blocks * K,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        transform=transform,
        sample_prior=sample_prior,
        K=K,
    )
    model.predict = predict  # type: ignore[attr-defined]
    return model
