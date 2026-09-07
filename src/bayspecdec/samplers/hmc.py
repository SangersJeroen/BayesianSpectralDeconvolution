"""
hmc.py — Fixed-trajectory Hamiltonian Monte Carlo kernel.

Implements:
  - Standard velocity Verlet (leapfrog) integrator.
  - Numerical central-difference gradient (no autodiff dependency).
  - Diagonal mass matrix / metric.
  - Metropolis correction with energy-error guard.
  - Optional dual-averaging step-size adaptation during warmup.
  - ``hmc_kernel_factory`` for drop-in use with ``ParallelTempering``.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from ..models import SpectralModel
from .metropolis import MetropolisState, MCMCKernel
from .adaptation import StepSizeAdaptation

Array = np.ndarray


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HMCConfig:
    """
    Configuration for the fixed-trajectory HMC kernel.

    Attributes
    ----------
    num_steps:
        Number of leapfrog steps *L* per transition.
    step_size:
        Initial leapfrog step size ε.  If ``adapt_step_size`` is True this is
        only the starting value; the dual-averaging scheme will tune it during
        warmup.
    target_accept:
        Desired mean Metropolis acceptance probability (used by dual averaging).
    max_energy_error:
        Transitions with ``|ΔH| > max_energy_error`` are treated as divergent
        and automatically rejected.  Divergences are counted but do not raise.
    adapt_step_size:
        If True, dual-averaging adaptation is active during warmup steps
        (``is_warmup=True`` in ``step``).
    """

    num_steps: int
    step_size: float
    target_accept: float = 0.8
    max_energy_error: float = 1000.0
    adapt_step_size: bool = True

    def __post_init__(self):
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if not (0.0 < self.target_accept < 1.0):
            raise ValueError("target_accept must be in (0, 1)")


# ---------------------------------------------------------------------------
# Gradient utility
# ---------------------------------------------------------------------------


def numerical_gradient(
    f: Callable[[Array], float],
    x: Array,
    eps: float = 1e-5,
) -> Array:
    """
    Central-difference numerical gradient of scalar function *f* at *x*.

    O(2·d) function evaluations.  Not for production throughput — use autodiff
    (JAX / torch) when that becomes available.
    """
    grad = np.empty_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        fp = f(x_plus)
        fm = f(x_minus)
        if not np.isfinite(fp) or not np.isfinite(fm):
            grad[i] = 0.0  # best-effort; caller should detect bad H
        else:
            grad[i] = (fp - fm) / (2.0 * eps)
    return grad


# ---------------------------------------------------------------------------
# Leapfrog integrator
# ---------------------------------------------------------------------------


def leapfrog(
    theta: Array,
    momentum: Array,
    step_size: float,
    n_steps: int,
    grad_potential_fn: Callable[[Array], Array],
    inverse_mass_matrix: Array,
) -> tuple[Array, Array]:
    """
    Velocity-Verlet (leapfrog) integrator.

    Performs ``n_steps`` full leapfrog steps starting from (θ, p).
    The returned (θ', p') is ready for a Metropolis accept/reject test.

    Parameters
    ----------
    inverse_mass_matrix:
        Diagonal inverse mass matrix stored as a 1-D array (M⁻¹ elementwise).
    """
    theta_t = theta.copy()
    p_t = momentum.copy()

    # Initial half-step on momentum
    p_t -= 0.5 * step_size * grad_potential_fn(theta_t)

    for i in range(n_steps):
        # Full position step
        theta_t = theta_t + step_size * (inverse_mass_matrix * p_t)

        # Full momentum step (skip final to merge with terminal half-step)
        grad = grad_potential_fn(theta_t)
        if i < n_steps - 1:
            p_t -= step_size * grad
        else:
            # Terminal half-step
            p_t -= 0.5 * step_size * grad

    return theta_t, p_t


# ---------------------------------------------------------------------------
# HMC kernel
# ---------------------------------------------------------------------------


class HamiltonianMonteCarlo(MCMCKernel):
    """
    Fixed-trajectory HMC kernel.

    Compatible with ``ParallelTempering`` via the ``MCMCKernel`` protocol.
    Accepts ``is_warmup`` to switch between the noisy and smoothed step size.
    """

    def __init__(
        self,
        model: SpectralModel,
        beta: float,
        rng: np.random.Generator,
        config: HMCConfig,
        mass_matrix: Optional[Array] = None,
    ):
        self.model = model
        self.beta = float(beta)
        self.rng = rng
        self.config = config

        ndim = model.parameterization.ndim
        self.mass_matrix = (
            np.ones(ndim)
            if mass_matrix is None
            else np.asarray(mass_matrix, dtype=float).copy()
        )
        self.inverse_mass_matrix = 1.0 / self.mass_matrix

        self.adaptation: Optional[StepSizeAdaptation] = (
            StepSizeAdaptation(
                initial_step_size=config.step_size,
                target_accept=config.target_accept,
            )
            if config.adapt_step_size
            else None
        )
        # Divergence counter (informational)
        self.n_divergent = 0

    # -- energy functions ----------------------------------------------------

    def potential_energy(self, theta: Array) -> float:
        """U(θ) = −log q_β(θ)."""
        val = self.model.log_tempered_target(theta, self.beta)
        return float(-val) if np.isfinite(val) else np.inf

    def grad_potential_energy(self, theta: Array) -> Array:
        return numerical_gradient(self.potential_energy, theta)

    def kinetic_energy(self, momentum: Array) -> float:
        """K(p) = ½ pᵀ M⁻¹ p."""
        return float(0.5 * np.dot(momentum, self.inverse_mass_matrix * momentum))

    def _sample_momentum(self) -> Array:
        """p ~ N(0, M)."""
        return self.rng.normal(0.0, np.sqrt(self.mass_matrix))

    # -- main step -----------------------------------------------------------

    def step(self, state: MetropolisState, is_warmup: bool = False) -> MetropolisState:
        # Select step size
        if self.adaptation is not None:
            eps = (
                self.adaptation.current_step_size()
                if is_warmup
                else self.adaptation.final_step_size()
            )
        else:
            eps = self.config.step_size

        # 1. Resample momentum
        p0 = self._sample_momentum()
        U0 = self.potential_energy(state.theta)
        K0 = self.kinetic_energy(p0)
        H0 = U0 + K0

        # 2. Leapfrog integration
        divergent = False
        try:
            theta1, p1 = leapfrog(
                state.theta,
                p0,
                eps,
                self.config.num_steps,
                self.grad_potential_energy,
                self.inverse_mass_matrix,
            )
            U1 = self.potential_energy(theta1)
            K1 = self.kinetic_energy(p1)
            H1 = U1 + K1
        except Exception:
            theta1, U1, K1, H1 = state.theta, np.inf, 0.0, np.inf

        # 3. Metropolis correction
        delta_H = H1 - H0
        if not np.isfinite(delta_H) or abs(delta_H) > self.config.max_energy_error:
            divergent = True
            alpha = 0.0
        else:
            log_alpha = min(0.0, -delta_H)
            alpha = np.exp(log_alpha)

        if divergent:
            self.n_divergent += 1

        # 4. Accept / reject
        accept = (not divergent) and (
            np.log(self.rng.random()) < np.log(alpha + 1e-300)
        )

        state.attempted += 1
        if accept:
            state.theta = theta1
            state.log_target = float(-U1)
            state.energy = self.model.energy(theta1)
            state.accepted += 1

        # 5. Update step-size adaptation during warmup
        if is_warmup and self.adaptation is not None:
            self.adaptation.update(alpha)

        return state


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def hmc_kernel_factory(
    model: SpectralModel,
    beta: float,
    rng: np.random.Generator,
    config: Optional[HMCConfig] = None,
) -> HamiltonianMonteCarlo:
    """Default factory for use with ``ParallelTempering``."""
    if config is None:
        config = HMCConfig(num_steps=10, step_size=0.01)
    return HamiltonianMonteCarlo(model, beta, rng, config)
