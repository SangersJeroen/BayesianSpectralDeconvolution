"""
tempering.py — Autodiff HMC with batched parallel tempering in JAX.

All L chains advance together: one ``vmap`` over temperatures per HMC step, a
``lax.scan`` over leapfrog steps and a second ``lax.scan`` over iterations, so a
whole run compiles to a single XLA program.

Replica exchange uses the usual odd/even sweep. Pairs inside a sweep do not
overlap, so all swap decisions are made at once and applied as one permutation of
the chain states. Step-size adaptation (dual averaging) is per chain, because each
temperature has its own optimal step size; a swap moves ``theta`` but never the
adaptation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..tempering import ExchangeResult
from .model import JaxModel

Array = jax.Array


@dataclass(frozen=True)
class PTConfig:
    """
    num_leapfrog:
        Leapfrog steps per transition (static: it fixes the scan length).
    init_step_size:
        Starting step size in z-space. Dual averaging tunes it per chain.
    target_accept:
        Dual-averaging target for the mean Metropolis acceptance probability.
    swap_every:
        HMC iterations between replica-exchange sweeps.
    max_energy_error:
        ``|dH|`` above this is a divergence and is rejected.
    """

    num_leapfrog: int = 10
    init_step_size: float = 0.1
    target_accept: float = 0.8
    swap_every: int = 10
    max_energy_error: float = 1000.0
    da_gamma: float = 0.05
    da_t0: float = 10.0
    da_kappa: float = 0.75

    def __post_init__(self):
        if self.num_leapfrog < 1:
            raise ValueError("num_leapfrog must be >= 1")
        if self.init_step_size <= 0.0:
            raise ValueError("init_step_size must be positive")
        if not (0.0 < self.target_accept < 1.0):
            raise ValueError("target_accept must be in (0, 1)")
        if self.swap_every < 1:
            raise ValueError("swap_every must be >= 1")


class JaxParallelTempering:
    """Batched HMC parallel tempering; ``run`` returns the numpy ``ExchangeResult``."""

    def __init__(
        self,
        model: JaxModel,
        betas,
        config: Optional[PTConfig] = None,
        inverse_mass: Optional[Array] = None,
    ):
        self.model = model
        self.betas = jnp.asarray(betas, dtype=float)
        if self.betas.size < 2:
            raise ValueError("Need at least two temperatures")
        self.L = int(self.betas.size)
        self.config = config or PTConfig()
        self.inverse_mass = (
            jnp.ones(model.ndim)
            if inverse_mass is None
            else jnp.asarray(inverse_mass, dtype=float)
        )
        self._run = jax.jit(self._run_impl, static_argnames=("burn_in", "samples"))
        self._to_theta = jax.jit(jax.vmap(jax.vmap(model.transform.to_theta)))

    # -- target in z-space ---------------------------------------------------

    def _parts(self, z: Array) -> tuple[Array, Array]:
        theta = self.model.transform.to_theta(z)
        ll = self.model.log_likelihood(theta)
        lp = self.model.log_prior(theta) + self.model.transform.log_jac(z)
        return ll, lp

    def _potential(self, z: Array, beta: Array):
        ll, lp = self._parts(z)
        return -(beta * ll + lp), (ll, lp)

    # -- one HMC transition for one chain ---------------------------------------

    def _hmc_step(self, key, z, ll, lp, beta, eps):
        cfg = self.config
        inv_m = self.inverse_mass
        vg = jax.value_and_grad(self._potential, has_aux=True)

        k_mom, k_acc = jax.random.split(key)
        p0 = jax.random.normal(k_mom, z.shape) / jnp.sqrt(inv_m)
        U0 = -(beta * ll + lp)
        H0 = U0 + 0.5 * jnp.sum(inv_m * p0**2)

        (_, _), g0 = vg(z, beta)
        p = p0 - 0.5 * eps * g0

        def body(carry, _):
            z, p, _g, _out = carry
            z = z + eps * inv_m * p
            out, g = vg(z, beta)
            return (z, p - eps * g, g, out), None

        init = (z, p, g0, (U0, (ll, lp)))
        (z1, p1, g1, (U1, (ll1, lp1))), _ = jax.lax.scan(
            body, init, None, length=cfg.num_leapfrog
        )
        p1 = p1 + 0.5 * eps * g1  # trade the last full momentum step for a half step

        dH = U1 + 0.5 * jnp.sum(inv_m * p1**2) - H0
        bad = ~jnp.isfinite(dH) | (jnp.abs(dH) > cfg.max_energy_error)
        alpha = jnp.where(bad, 0.0, jnp.exp(jnp.minimum(0.0, -dH)))
        accept = (~bad) & (jnp.log(jax.random.uniform(k_acc)) < jnp.log(alpha + 1e-300))

        return (
            jnp.where(accept, z1, z),
            jnp.where(accept, ll1, ll),
            jnp.where(accept, lp1, lp),
            alpha,
            accept,
        )

    # -- replica exchange -------------------------------------------------------

    def _swap(self, key, z, ll, lp, parity):
        L, betas = self.L, self.betas
        idx = jnp.arange(L - 1)
        # log v = (beta_{l+1} - beta_l) (ll_l - ll_{l+1}), as in the numpy sampler.
        log_v = (betas[1:] - betas[:-1]) * (ll[:-1] - ll[1:])
        active = (idx % 2) == parity
        accept = active & (jnp.log(jax.random.uniform(key, (L - 1,))) < jnp.minimum(0.0, log_v))
        # Out-of-range index L is dropped, so only accepted pairs write to ``perm``.
        perm = jnp.arange(L)
        perm = perm.at[jnp.where(accept, idx, L)].set(idx + 1, mode="drop")
        perm = perm.at[jnp.where(accept, idx + 1, L)].set(idx, mode="drop")
        return z[perm], ll[perm], lp[perm], active, accept

    # -- dual averaging (vectorised over chains) ------------------------------------

    def _da_update(self, da, alpha):
        cfg = self.config
        log_eps, log_eps_bar, h_bar, t, mu = da
        eta = 1.0 / (t + cfg.da_t0)
        h_bar = (1.0 - eta) * h_bar + eta * (cfg.target_accept - alpha)
        log_eps = mu - (jnp.sqrt(t) / cfg.da_gamma) * h_bar
        eta_bar = t ** (-cfg.da_kappa)
        log_eps_bar = eta_bar * log_eps + (1.0 - eta_bar) * log_eps_bar
        return (log_eps, log_eps_bar, h_bar, t + 1.0, mu)

    # -- full run -------------------------------------------------------------------

    def _iteration(self, carry, t, adapt: bool):
        key, z, ll, lp, da, ex_att, ex_acc = carry
        key, k_hmc, k_swap = jax.random.split(key, 3)

        log_eps = da[0] if adapt else da[1]
        keys = jax.random.split(k_hmc, self.L)
        z, ll, lp, alpha, accepted = jax.vmap(self._hmc_step)(
            keys, z, ll, lp, self.betas, jnp.exp(log_eps)
        )
        if adapt:
            da = self._da_update(da, alpha)

        swap_now = (t % self.config.swap_every) == 0
        parity = (t // self.config.swap_every) % 2
        z_s, ll_s, lp_s, active, swapped = self._swap(k_swap, z, ll, lp, parity)
        z = jnp.where(swap_now, z_s, z)
        ll = jnp.where(swap_now, ll_s, ll)
        lp = jnp.where(swap_now, lp_s, lp)
        ex_att = ex_att + swap_now * active
        ex_acc = ex_acc + swap_now * swapped

        return (key, z, ll, lp, da, ex_att, ex_acc), (z, ll, accepted)

    def _run_impl(self, key, z0, burn_in: int, samples: int):
        L = self.L
        ll0, lp0 = jax.vmap(self._parts)(z0)
        eps0 = self.config.init_step_size
        da = (
            jnp.full(L, jnp.log(eps0)),
            jnp.full(L, jnp.log(eps0)),
            jnp.zeros(L),
            jnp.asarray(1.0),
            jnp.full(L, jnp.log(10.0 * eps0)),
        )
        zeros = jnp.zeros(L - 1, dtype=int)
        carry = (key, z0, ll0, lp0, da, zeros, zeros)

        carry, _ = jax.lax.scan(
            partial(self._iteration, adapt=True), carry, jnp.arange(1, burn_in + 1)
        )
        # Exchange statistics only count the sampling phase.
        carry = carry[:5] + (zeros, zeros)
        carry, (z_tr, ll_tr, acc_tr) = jax.lax.scan(
            partial(self._iteration, adapt=False),
            carry,
            jnp.arange(burn_in + 1, burn_in + samples + 1),
        )
        _, _, _, _, da, ex_att, ex_acc = carry
        return z_tr, ll_tr, acc_tr, ex_att, ex_acc, jnp.exp(da[1])

    def run(self, burn_in: int, samples: int, seed: int = 0) -> ExchangeResult:
        """Warm up (adapting step sizes), then record ``samples`` iterations."""
        key_init, key_run = jax.random.split(jax.random.PRNGKey(seed))
        theta0 = jax.vmap(self.model.sample_prior)(jax.random.split(key_init, self.L))
        z0 = jax.vmap(self.model.transform.to_z)(theta0)

        z_tr, ll_tr, acc_tr, ex_att, ex_acc, eps = self._run(
            key_run, z0, burn_in=burn_in, samples=samples
        )
        theta_tr = np.asarray(self._to_theta(z_tr))  # (samples, L, d)
        ll_tr = np.asarray(ll_tr)
        ex_att, ex_acc = np.asarray(ex_att), np.asarray(ex_acc)

        self.final_step_sizes = np.asarray(eps)
        return ExchangeResult(
            beta=np.asarray(self.betas),
            samples_by_temperature=[theta_tr[:, l] for l in range(self.L)],
            energy_trace_by_temperature=[-ll_tr[:, l] for l in range(self.L)],
            log_likelihood_trace_by_temperature=[ll_tr[:, l] for l in range(self.L)],
            within_acceptance=np.asarray(acc_tr).mean(axis=0),
            exchange_acceptance=ex_acc / np.maximum(ex_att, 1),
        )
