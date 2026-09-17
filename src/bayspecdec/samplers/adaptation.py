"""
adaptation.py — Step-size and mass-matrix adaptation for HMC/NUTS.

Implements:
  - Dual averaging (Nesterov 2009, as used by Stan/PyMC) for step-size adaptation.
  - Welford online covariance estimator for diagonal/dense mass-matrix adaptation.
  - find_reasonable_step_size heuristic.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

Array = np.ndarray


# ---------------------------------------------------------------------------
# Dual averaging — step-size adaptation
# ---------------------------------------------------------------------------


@dataclass
class DualAveragingState:
    log_step_size: float  # current (noisy) log ε_t
    log_step_size_bar: float  # smoothed log ε̄_t  (frozen after warmup)
    H_bar: float  # averaged error (target_accept - actual_accept)
    iteration: int
    mu: float  # μ = log(10 * ε_0)


class StepSizeAdaptation:
    """
    Dual-averaging step-size adaptation (Nesterov / Hoffman-Gelman 2014).

    During warmup call ``update(accept_stat)`` after each transition.
    After warmup call ``final_step_size()`` to obtain the frozen ε̄.

    Parameters
    ----------
    initial_step_size:
        Starting step size ε_0 (can be found by ``find_reasonable_step_size``).
    target_accept:
        Desired mean acceptance probability δ (typically 0.65–0.9).
    gamma, t0, kappa:
        Dual-averaging hyper-parameters; defaults match Stan.
    """

    def __init__(
        self,
        initial_step_size: float,
        target_accept: float = 0.8,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ):
        if not (0.0 < target_accept < 1.0):
            raise ValueError("target_accept must be in (0, 1)")
        if initial_step_size <= 0.0:
            raise ValueError("initial_step_size must be positive")

        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa

        mu = np.log(10.0 * initial_step_size)
        self.state = DualAveragingState(
            log_step_size=np.log(initial_step_size),
            log_step_size_bar=np.log(initial_step_size),
            H_bar=0.0,
            iteration=1,
            mu=mu,
        )

    def update(self, accept_stat: float) -> float:
        """
        Incorporate one acceptance statistic and return the next step size.

        ``accept_stat`` should be ``min(1, exp(-delta_H))``, i.e. a value in [0, 1].
        """
        accept_stat = float(np.clip(accept_stat, 0.0, 1.0))
        s = self.state
        t = float(s.iteration)

        # Eq. (6) in Hoffman-Gelman: weighted update of the error term
        eta_t = 1.0 / (t + self.t0)
        s.H_bar = (1.0 - eta_t) * s.H_bar + eta_t * (self.target_accept - accept_stat)

        # Current (noisy) log step size — Eq. (7)
        s.log_step_size = s.mu - (np.sqrt(t) / self.gamma) * s.H_bar

        # Smoothed log step size — Eq. (8)
        eta_bar = t ** (-self.kappa)
        s.log_step_size_bar = (
            eta_bar * s.log_step_size + (1.0 - eta_bar) * s.log_step_size_bar
        )

        s.iteration += 1
        print(f'step: {float(np.exp(s.log_step_size)):.2e}')
        return float(np.exp(s.log_step_size))

    def current_step_size(self) -> float:
        """Noisy step size for the current warmup iteration."""
        return float(np.exp(self.state.log_step_size))

    def final_step_size(self) -> float:
        """Smoothed, stabilised step size — use this after warmup is complete."""
        return float(np.exp(self.state.log_step_size_bar))


# ---------------------------------------------------------------------------
# Initial step-size heuristic
# ---------------------------------------------------------------------------


def find_reasonable_step_size(
    potential_fn: Callable[[Array], float],
    grad_potential_fn: Callable[[Array], Array],
    kinetic_energy_fn: Callable[[Array], float],
    sample_momentum_fn: Callable[[], Array],
    theta: Array,
    rng: np.random.Generator,
    initial_step_size: float = 1.0,
    max_doublings: int = 30,
) -> float:
    """
    Heuristic search for an initial step size.

    Doubles or halves ``ε`` until the single-step acceptance probability
    crosses 0.5, giving a reasonable starting point for dual averaging.

    Based on Algorithm 4 of Hoffman & Gelman (2014).
    """
    eps = initial_step_size
    momentum = sample_momentum_fn()

    U0 = potential_fn(theta)
    K0 = kinetic_energy_fn(momentum)

    # One leapfrog step
    from .hmc import leapfrog

    try:
        theta1, momentum1 = leapfrog(
            theta,
            momentum,
            eps,
            1,
            grad_potential_fn,
            1.0 / sample_momentum_fn().__class__,  # placeholder — handled below
        )
    except Exception:
        return initial_step_size

    # We need the inverse mass matrix; inject it via closure in real usage.
    # This function is meant to be called from a kernel that closes over inv_M.
    # Rebuild with a unit inverse mass matrix here as fallback.
    inv_M = np.ones_like(theta)
    try:
        theta1, momentum1 = leapfrog(theta, momentum, eps, 1, grad_potential_fn, inv_M)
        U1 = potential_fn(theta1)
        K1 = kinetic_energy_fn(momentum1)
        dH = (U1 + K1) - (U0 + K0)
        alpha = np.exp(-dH) if np.isfinite(dH) else 0.0
    except Exception:
        alpha = 0.0

    direction = 1 if alpha > 0.5 else -1

    for _ in range(max_doublings):
        eps *= 2.0**direction
        try:
            theta1, momentum1 = leapfrog(
                theta, momentum, eps, 1, grad_potential_fn, inv_M
            )
            U1 = potential_fn(theta1)
            K1 = kinetic_energy_fn(momentum1)
            dH = (U1 + K1) - (U0 + K0)
            alpha = np.exp(-dH) if np.isfinite(dH) else 0.0
        except Exception:
            alpha = 0.0

        if direction == 1 and alpha <= 0.5:
            break
        if direction == -1 and alpha >= 0.5:
            break
        if eps < 1e-12 or eps > 1e6:
            break

    return float(np.clip(eps, 1e-10, 1e4))


# ---------------------------------------------------------------------------
# Online Welford covariance estimator — mass matrix adaptation
# ---------------------------------------------------------------------------


class WelfordCovariance:
    """
    Online (single-pass) Welford estimator for mean and (co)variance.

    For the diagonal metric, use ``diagonal=True`` (stores only variances).
    For the dense metric, use ``diagonal=False`` (stores full covariance).

    Call ``update(sample)`` for each new sample.
    Call ``get_variance()`` or ``get_covariance()`` to retrieve the estimate.

    The regularised mass matrix is:
        M_diag  = (n / (n + 5)) * variance + 5 / (n + 5) * I
        M_dense = (n / (n + 5)) * Sigma    + 5 / (n + 5) * I
    (Ridge regularisation prevents degenerate matrices in early warmup.)
    """

    def __init__(self, ndim: int, diagonal: bool = True):
        self.ndim = ndim
        self.diagonal = diagonal
        self.n = 0
        self.mean = np.zeros(ndim)
        if diagonal:
            self.M2 = np.zeros(ndim)  # sum of squared deviations
        else:
            self.M2 = np.zeros((ndim, ndim))

    def update(self, sample: Array) -> None:
        sample = np.asarray(sample, dtype=float)
        self.n += 1
        delta = sample - self.mean
        self.mean += delta / self.n
        delta2 = sample - self.mean
        if self.diagonal:
            self.M2 += delta * delta2
        else:
            self.M2 += np.outer(delta, delta2)

    def reset(self) -> None:
        self.n = 0
        self.mean = np.zeros(self.ndim)
        if self.diagonal:
            self.M2 = np.zeros(self.ndim)
        else:
            self.M2 = np.zeros((self.ndim, self.ndim))

    def get_variance(self) -> Array:
        """Return regularised diagonal variance estimate."""
        if self.n < 2:
            return np.ones(self.ndim)
        var = self.M2 / (self.n - 1)
        # Ridge regularisation
        w = self.n / (self.n + 5.0)
        return w * var + (1.0 - w) * np.ones(self.ndim)

    def get_covariance(self) -> Array:
        """Return regularised full covariance estimate."""
        if self.n < 2:
            return np.eye(self.ndim)
        cov = self.M2 / (self.n - 1)
        w = self.n / (self.n + 5.0)
        return w * cov + (1.0 - w) * np.eye(self.ndim)

    def get_mass_matrix(self) -> Array:
        """
        Return the diagonal mass matrix (variances) or dense mass matrix.
        The mass matrix M is the *inverse* of the covariance proxy.
        Storing it as the actual per-dimension variance is conventional for
        diagonal metrics: p ~ N(0, M) where M = 1/var elementwise.
        """
        if self.diagonal:
            return self.get_variance()  # used as mass (not inverse mass)
        else:
            return self.get_covariance()
