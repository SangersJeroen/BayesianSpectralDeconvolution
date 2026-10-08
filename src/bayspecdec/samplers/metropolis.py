import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol
from ..models import SpectralModel

Array = np.ndarray


@dataclass
class MetropolisState:
    theta: Array
    log_target: float
    energy: float
    accepted: int = 0
    attempted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempted if self.attempted else np.nan


class MCMCKernel(Protocol):
    def step(
        self, state: MetropolisState, is_warmup: bool = False
    ) -> MetropolisState: ...


class RandomWalkMetropolis:
    """
    Gaussian random-walk Metropolis in the unconstrained coordinates ``z``.

    The proposal is symmetric in ``z``; the target density in ``z`` carries the Jacobian of the
    map ``z -> theta``, so the acceptance ratio includes ``log|d theta/d z|`` at the proposal
    and the current state. ``use_log_jacobian=False`` drops it (only valid for an identity
    transform).
    """

    def __init__(
        self,
        model: SpectralModel,
        beta: float,
        rng: np.random.Generator,
        proposal_scales: Optional[Array] = None,
        use_log_jacobian: bool = True,
    ):
        self.model = model
        self.beta = float(beta)
        self.rng = rng
        self.use_log_jacobian = bool(use_log_jacobian)
        ndim = getattr(model.parameterization, "ndim", 1)
        self.proposal_scales = (
            np.full(ndim, 0.1)
            if proposal_scales is None
            else np.asarray(proposal_scales, dtype=float)
        )

    def step(self, state: MetropolisState, is_warmup: bool = False) -> MetropolisState:
        param = self.model.parameterization
        proposal = param.from_z(
            param.to_z(state.theta) + self.rng.normal(0.0, self.proposal_scales)
        )
        proposal_log_target = self.model.log_tempered_target(proposal, self.beta)

        log_alpha = proposal_log_target - state.log_target
        if self.use_log_jacobian and np.isfinite(proposal_log_target):
            log_alpha += param.log_jacobian(proposal) - param.log_jacobian(state.theta)
        if np.isnan(log_alpha):
            log_alpha = -np.inf
        accept = np.log(self.rng.random()) < min(0.0, float(log_alpha))

        state.attempted += 1
        if accept:
            state.theta = proposal
            state.log_target = proposal_log_target
            state.energy = self.model.energy(proposal)
            state.accepted += 1
        return state


def metropolis_kernel_factory(
    model: SpectralModel,
    beta: float,
    rng: np.random.Generator,
    proposal_scales: Optional[Array] = None,
    use_log_jacobian: bool = True,
) -> RandomWalkMetropolis:
    return RandomWalkMetropolis(model, beta, rng, proposal_scales, use_log_jacobian)
