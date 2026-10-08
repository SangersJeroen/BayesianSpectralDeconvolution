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

from .likelihoods import gaussian_log_likelihood, gaussian_noise_likelihood
from .priors import as_prior

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
    predict:
        ``theta -> (n,)`` noiseless model spectrum (optional; set by ``make_spectral_model``).
    """

    ndim: int
    log_likelihood: Callable[[Array], Array]
    log_prior: Callable[[Array], Array]
    transform: BoxTransform
    sample_prior: Callable[[Array], Array]
    K: Optional[int] = None
    n_noise: int = 0
    n_background: int = 0
    predict: Optional[Callable[[Array], Array]] = None

    def split(self, theta: Array) -> tuple[Array, Array]:
        """``theta -> (physical parameters, noise hyperparameters)`` along the last axis."""
        cut = self.ndim - self.n_noise
        return theta[..., :cut], theta[..., cut:]

    def split_background(self, theta: Array) -> tuple[Array, Array, Array]:
        """``theta -> (peak parameters, background parameters, noise hyperparameters)``."""
        phys, noise = self.split(theta)
        cut = phys.shape[-1] - self.n_background
        return phys[..., :cut], phys[..., cut:], noise

    def log_posterior(self, theta: Array) -> Array:
        return self.log_likelihood(theta) + self.log_prior(theta)


# ---------------------------------------------------------------------------
# Spectral-model convenience builder
# ---------------------------------------------------------------------------


def make_spectral_model(
    x: np.ndarray,
    y: np.ndarray,
    basis_fn: Callable[[Array, Array], Array],
    K: int,
    amplitude_prior,
    basis_priors: Sequence,
    sigma2: Optional[float] = None,
    likelihood: Optional[Callable[[Array, Array], Array]] = None,
    noise_prior=None,
    noise_likelihood: Optional[Callable[[Array, Array, Array], Array]] = None,
    background_fn: Optional[Callable[[Array, Array], Array]] = None,
    background_priors: Optional[Sequence] = None,
) -> JaxModel:
    """
    ``y ~ sum_k a_k * basis_fn(x, params)[k]`` with independent priors per parameter block.

    theta layout: ``[a_1..a_K, p1_1..p1_K, p2_1..p2_K, ..., s_1..s_m]``: one block of K values for
    the amplitudes and one per entry of ``basis_priors``, then ``m`` noise hyperparameters when
    the noise is sampled. Each block shares one prior, given as a ``UniformPrior`` /
    ``GammaPrior`` / ... (or any ``JaxPrior``) or a ``(lower, upper)`` tuple meaning uniform.
    ``basis_fn`` receives the basis blocks stacked as ``(n_basis_params, K)`` and returns ``(K, n)``.

    An optional **background** is added to the peaks: pass ``background_fn(x, params) -> (n,)`` and one
    prior per entry of ``params`` in ``background_priors`` (see ``background.py`` for ready-made
    functions, e.g. ``arctan_step_background`` with six priors). Its parameters sit between the
    basis blocks and the noise hyperparameters::

        [a_1..a_K, p1_1..p1_K, ..., b_1..b_m, s_1..s_m']

    ``model.split`` still returns ``(everything but noise, noise)``; use ``model.split_background``
    to also separate the ``m`` background parameters.

    Noise is either **fixed** or **sampled**; choose with the arguments (exactly one mode):

    * fixed: pass ``sigma2`` (Gaussian) or a custom ``likelihood(y, pred)``;
    * sampled: pass ``noise_prior`` — a prior (or a sequence of priors) over the noise
      hyperparameter(s), appended to ``theta`` — and optionally ``noise_likelihood(y, pred, s)``
      with ``s`` of shape ``(m,)``. The default is Gaussian noise with ``s = [log sigma^2]``,
      so ``noise_prior=log_scale_prior(1e-8, 1e-1)`` alone samples ``sigma^2`` log-uniformly.
      Use ``model.split(theta)`` to separate the noise columns from the physical ones.

    The likelihood stays fully normalised in the sampled mode (the ``-n/2 log sigma^2`` term is
    what stops sigma^2 growing without bound), so ``log Z`` is marginalised over the noise and is
    comparable across ``K`` as long as the noise prior is the same.
    """
    sampled = noise_prior is not None
    if sampled:
        if sigma2 is not None or likelihood is not None:
            raise ValueError(
                "Sampled noise: do not pass sigma2 or likelihood, use noise_likelihood"
            )
        noise_priors = (
            [as_prior(p) for p in noise_prior]
            if isinstance(noise_prior, (list, tuple))
            and not (len(noise_prior) == 2 and all(np.isscalar(v) for v in noise_prior))
            else [as_prior(noise_prior)]
        )
        if noise_likelihood is None:
            if len(noise_priors) != 1:
                raise ValueError(
                    "Give noise_likelihood when sampling more than one noise parameter"
                )
            noise_likelihood = gaussian_noise_likelihood()
    else:
        if noise_likelihood is not None:
            raise ValueError("noise_likelihood needs a noise_prior")
        if (sigma2 is None) == (likelihood is None):
            raise ValueError(
                "Pass exactly one of sigma2 or likelihood, or a noise_prior"
            )
        if likelihood is None:
            likelihood = gaussian_log_likelihood(sigma2)
        noise_priors = []

    if (background_fn is None) != (background_priors is None):
        raise ValueError("Pass background_fn and background_priors together")
    bg_priors = [as_prior(p) for p in (background_priors or [])]
    n_bg = len(bg_priors)
    if background_fn is not None:
        expected = getattr(background_fn, "n_params", n_bg)
        if n_bg == 0 or n_bg != expected:
            raise ValueError(
                f"background_fn takes {expected} parameters but {n_bg} background_priors were given"
            )

    priors = [as_prior(amplitude_prior)] + [as_prior(p) for p in basis_priors]
    n_blocks = len(priors)
    n_peak = n_blocks * K
    n_phys = n_peak + n_bg
    n_noise = len(noise_priors)
    transform = BoxTransform(
        np.concatenate(
            [
                np.repeat([p.lower for p in priors], K),
                [p.lower for p in bg_priors],
                [p.lower for p in noise_priors],
            ]
        ),
        np.concatenate(
            [
                np.repeat([p.upper for p in priors], K),
                [p.upper for p in bg_priors],
                [p.upper for p in noise_priors],
            ]
        ),
    )

    xj = jnp.asarray(x, dtype=float)
    yj = jnp.asarray(y, dtype=float)

    def predict(theta: Array) -> Array:
        blocks = theta[:n_peak].reshape(n_blocks, K)
        pred = blocks[0] @ basis_fn(xj, blocks[1:])
        if background_fn is not None:
            pred = pred + background_fn(xj, theta[n_peak:n_phys])
        return pred

    def log_likelihood(theta: Array) -> Array:
        if sampled:
            return noise_likelihood(yj, predict(theta), theta[n_phys:])
        return likelihood(yj, predict(theta))

    def log_prior(theta: Array) -> Array:
        blocks = theta[:n_peak].reshape(n_blocks, K)
        total = sum(jnp.sum(p.log_prob(blocks[i])) for i, p in enumerate(priors))
        for j, p in enumerate(bg_priors):
            total = total + jnp.sum(p.log_prob(theta[n_peak + j]))
        for j, p in enumerate(noise_priors):
            total = total + jnp.sum(p.log_prob(theta[n_phys + j]))
        return total

    def sample_prior(key: Array) -> Array:
        keys = jax.random.split(key, n_blocks + n_bg + n_noise)
        parts = [p.sample(k, (K,)) for p, k in zip(priors, keys)]
        parts += [
            p.sample(k, (1,))
            for p, k in zip(bg_priors, keys[n_blocks : n_blocks + n_bg])
        ]
        parts += [
            p.sample(k, (1,)) for p, k in zip(noise_priors, keys[n_blocks + n_bg :])
        ]
        return jnp.concatenate(parts)

    model = JaxModel(
        ndim=n_phys + n_noise,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        transform=transform,
        sample_prior=sample_prior,
        K=K,
        n_noise=n_noise,
        n_background=n_bg,
        predict=predict,
    )
    return model
