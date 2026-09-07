"""
nuts.py — No-U-Turn Sampler (NUTS) kernel.

Implements Algorithm 3 from Hoffman & Gelman (2014) — the "efficient NUTS"
variant with slice sampling and multinomial-like candidate selection.

Key design points
-----------------
- Builds a binary tree via recursion; stops on U-turn or divergence.
- Uses the generalized U-turn criterion (dot-product test on both ends).
- Tracks divergences, tree depth, and leapfrog step counts.
- Integrates with ``StepSizeAdaptation`` (dual averaging) during warmup.
- Uses the same ``MetropolisState`` as Metropolis/HMC for compatibility with
  ``ParallelTempering``.
- No JAX dependency; uses numerical central-difference gradients.

Algorithmic variant
-------------------
Slice-based original NUTS (Hoffman & Gelman 2014, Algorithm 3) with the
biased progressive proposal update ("naive NUTS").  This is the simplest
correct variant.  A multinomial-tree NUTS formulation (Betancourt 2017) is
mathematically superior but more complex; it can be swapped in as a future
``nuts.py`` replacement without changing external interfaces.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..models import SpectralModel
from .metropolis import MetropolisState, MCMCKernel
from .adaptation import StepSizeAdaptation
from .hmc import numerical_gradient, leapfrog

Array = np.ndarray


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NUTSConfig:
    """
    Configuration for the NUTS kernel.

    Attributes
    ----------
    step_size:
        Initial leapfrog step size.  Tuned by dual averaging during warmup.
    target_accept:
        Target mean acceptance probability for dual averaging (typically 0.65–0.9).
    max_tree_depth:
        Maximum binary-tree depth.  The trajectory contains at most
        ``2^max_tree_depth`` leapfrog steps.  Hitting this limit is a diagnostic
        signal that the geometry is difficult or the step size is too small.
    max_energy_error:
        Absolute Hamiltonian error threshold above which a transition is
        declared divergent.
    adapt_step_size:
        Enable dual-averaging step-size adaptation during warmup.
    """

    step_size: float
    target_accept: float = 0.8
    max_tree_depth: int = 10
    max_energy_error: float = 1000.0
    adapt_step_size: bool = True

    def __post_init__(self):
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if not (0.0 < self.target_accept < 1.0):
            raise ValueError("target_accept must be in (0, 1)")
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be >= 1")


# ---------------------------------------------------------------------------
# Tree-building sub-structures
# ---------------------------------------------------------------------------


@dataclass
class _LeafResult:
    """Return type from the base case of ``_build_tree``."""

    theta_minus: Array
    p_minus: Array
    theta_plus: Array
    p_plus: Array
    theta_proposal: Array
    n_valid: int  # number of slice-valid states in this sub-tree
    divergent: bool  # did any step diverge?
    sum_accept: float  # cumulative accept prob for adaptation
    n_accept: int  # count for the above


# ---------------------------------------------------------------------------
# U-turn criterion
# ---------------------------------------------------------------------------


def is_uturn(
    theta_minus: Array,
    theta_plus: Array,
    p_minus: Array,
    p_plus: Array,
    inv_M: Array,
) -> bool:
    """
    Generalized U-turn criterion (Betancourt 2017 / Stan Reference Manual).

    Returns True if the trajectory is turning back on itself.
    Both endpoint conditions must be checked.
    """
    delta = theta_plus - theta_minus
    return (
        float(np.dot(delta, inv_M * p_minus)) < 0.0
        or float(np.dot(delta, inv_M * p_plus)) < 0.0
    )


# ---------------------------------------------------------------------------
# NUTS kernel
# ---------------------------------------------------------------------------


class NoUTurnSampler(MCMCKernel):
    """
    NUTS kernel (slice-based, Algorithm 3 of Hoffman & Gelman 2014).

    Compatible with ``ParallelTempering`` via the ``MCMCKernel`` protocol.
    """

    def __init__(
        self,
        model: SpectralModel,
        beta: float,
        rng: np.random.Generator,
        config: NUTSConfig,
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

        # Diagnostic counters
        self.n_divergent = 0
        self.tree_depth_last = 0

    # -- energy helpers ------------------------------------------------------

    def potential_energy(self, theta: Array) -> float:
        val = self.model.log_tempered_target(theta, self.beta)
        return float(-val) if np.isfinite(val) else np.inf

    def grad_potential_energy(self, theta: Array) -> Array:
        return numerical_gradient(self.potential_energy, theta)

    def kinetic_energy(self, p: Array) -> float:
        return float(0.5 * np.dot(p, self.inverse_mass_matrix * p))

    def hamiltonian(self, theta: Array, p: Array) -> float:
        return self.potential_energy(theta) + self.kinetic_energy(p)

    def _sample_momentum(self) -> Array:
        return self.rng.normal(0.0, np.sqrt(self.mass_matrix))

    # -- recursive tree builder ----------------------------------------------

    def _build_tree(
        self,
        theta: Array,
        p: Array,
        log_u: float,
        direction: int,  # +1 forward, -1 backward
        depth: int,
        step_size: float,
        H0: float,
    ) -> _LeafResult:
        """
        Recursively build a balanced binary subtree of depth ``depth``.

        Implements the recursive doubling strategy of Algorithm 3.
        """
        if depth == 0:
            # ---------- Base case: one leapfrog step ----------
            try:
                theta_new, p_new = leapfrog(
                    theta,
                    p,
                    direction * step_size,
                    1,
                    self.grad_potential_energy,
                    self.inverse_mass_matrix,
                )
                H_new = self.hamiltonian(theta_new, p_new)
                divergent = (
                    not np.isfinite(H_new)
                    or abs(H_new - H0) > self.config.max_energy_error
                )
            except Exception:
                theta_new = theta.copy()
                p_new = p.copy()
                H_new = np.inf
                divergent = True

            # Slice criterion: valid if u ≤ exp(−H) ⟺ log_u ≤ −H
            n_valid = int((not divergent) and (log_u <= -H_new))

            # Acceptance probability for step-size adaptation.
            # Clamp delta_H before exp to prevent float overflow.
            delta_H = H_new - H0
            if divergent or not np.isfinite(delta_H):
                alpha = 0.0
            else:
                alpha = float(min(1.0, np.exp(float(np.clip(-delta_H, -500.0, 500.0)))))

            return _LeafResult(
                theta_minus=theta_new,
                p_minus=p_new,
                theta_plus=theta_new,
                p_plus=p_new,
                theta_proposal=theta_new,
                n_valid=n_valid,
                divergent=divergent,
                sum_accept=alpha,
                n_accept=1,
            )

        # ---------- Recursive case: build two subtrees ----------
        # First half
        r1 = self._build_tree(theta, p, log_u, direction, depth - 1, step_size, H0)

        if r1.divergent:
            return r1  # Short-circuit: already diverged

        # Grow in the same direction from the frontier endpoint
        if direction == -1:
            frontier_theta, frontier_p = r1.theta_minus, r1.p_minus
        else:
            frontier_theta, frontier_p = r1.theta_plus, r1.p_plus

        r2 = self._build_tree(
            frontier_theta,
            frontier_p,
            log_u,
            direction,
            depth - 1,
            step_size,
            H0,
        )

        # Merge the two subtrees
        n_total = r1.n_valid + r2.n_valid

        # Progressive proposal: accept r2's proposal proportionally
        if n_total > 0 and self.rng.random() < r2.n_valid / n_total:
            proposal = r2.theta_proposal
        else:
            proposal = r1.theta_proposal

        # Update frontier endpoints
        if direction == -1:
            theta_minus = r2.theta_minus
            p_minus = r2.p_minus
            theta_plus = r1.theta_plus
            p_plus = r1.p_plus
        else:
            theta_minus = r1.theta_minus
            p_minus = r1.p_minus
            theta_plus = r2.theta_plus
            p_plus = r2.p_plus

        # U-turn check on the merged subtree
        uturn = is_uturn(
            theta_minus, theta_plus, p_minus, p_plus, self.inverse_mass_matrix
        )

        return _LeafResult(
            theta_minus=theta_minus,
            p_minus=p_minus,
            theta_plus=theta_plus,
            p_plus=p_plus,
            theta_proposal=proposal,
            n_valid=n_total,
            divergent=r2.divergent or uturn,
            sum_accept=r1.sum_accept + r2.sum_accept,
            n_accept=r1.n_accept + r2.n_accept,
        )

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

        # Resample momentum
        p0 = self._sample_momentum()
        H0 = self.hamiltonian(state.theta, p0)

        # Sample slice variable u ~ Uniform(0, exp(−H0))
        # ⟺ log_u = log(U) − H0  where U ~ Uniform(0, 1)
        if not np.isfinite(H0):
            # Can't start from a degenerate state; skip transition
            state.attempted += 1
            return state
        log_u = float(np.log(self.rng.random())) - H0

        # Initialise doubly-linked tree endpoints at the current state
        theta_minus = state.theta.copy()
        p_minus = p0.copy()
        theta_plus = state.theta.copy()
        p_plus = p0.copy()

        theta_new = state.theta.copy()
        n_valid = 1  # current state is always slice-valid

        sum_accept = 0.0
        n_accept_total = 0
        stop = False
        depth = 0

        state.attempted += 1

        while not stop and depth < self.config.max_tree_depth:
            # Choose direction: −1 (backward) or +1 (forward)
            direction = self.rng.choice([-1, 1])

            if direction == -1:
                result = self._build_tree(
                    theta_minus,
                    p_minus,
                    log_u,
                    direction,
                    depth,
                    eps,
                    H0,
                )
                theta_minus = result.theta_minus
                p_minus = result.p_minus
            else:
                result = self._build_tree(
                    theta_plus,
                    p_plus,
                    log_u,
                    direction,
                    depth,
                    eps,
                    H0,
                )
                theta_plus = result.theta_plus
                p_plus = result.p_plus

            # Biased progressive update: accept new proposal
            if not result.divergent and n_valid > 0:
                accept_prob = min(1.0, result.n_valid / n_valid)
                if self.rng.random() < accept_prob:
                    theta_new = result.theta_proposal

            n_valid += result.n_valid
            sum_accept += result.sum_accept
            n_accept_total += result.n_accept

            # Stop on divergence or global U-turn
            stop = result.divergent or is_uturn(
                theta_minus,
                theta_plus,
                p_minus,
                p_plus,
                self.inverse_mass_matrix,
            )

            if result.divergent:
                self.n_divergent += 1

            depth += 1

        self.tree_depth_last = depth

        # Update state
        if not np.array_equal(theta_new, state.theta):
            new_U = self.potential_energy(theta_new)
            if np.isfinite(new_U):
                state.theta = theta_new
                state.log_target = float(-new_U)
                state.energy = self.model.energy(theta_new)
                state.accepted += 1

        # Update step-size adaptation
        if is_warmup and self.adaptation is not None:
            mean_alpha = sum_accept / max(n_accept_total, 1)
            self.adaptation.update(float(np.clip(mean_alpha, 0.0, 1.0)))

        return state


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def nuts_kernel_factory(
    model: SpectralModel,
    beta: float,
    rng: np.random.Generator,
    config: Optional[NUTSConfig] = None,
) -> NoUTurnSampler:
    """Default factory for use with ``ParallelTempering``."""
    if config is None:
        config = NUTSConfig(step_size=0.01)
    return NoUTurnSampler(model, beta, rng, config)
