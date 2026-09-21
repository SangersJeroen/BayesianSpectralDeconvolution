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
    Simple Gaussian random-walk Metropolis sampler.
    """

    def __init__(
        self,
        model: SpectralModel,
        beta: float,
        rng: np.random.Generator,
        proposal_scales: Optional[Array] = None,
    ):
        self.model = model
        self.beta = float(beta)
        self.rng = rng
        self.proposal_scales = np.asarray(proposal_scales, dtype=float)

    def step(self, state: MetropolisState, is_warmup: bool = False) -> MetropolisState:
        to_z, from_z = (
            self.model.parameterization.to_z,
            self.model.parameterization.from_z,
        )
        proposal = from_z(
            to_z(state.theta) + self.rng.normal(0.0, self.proposal_scales)
        )
        proposal_log_target = self.model.log_tempered_target(proposal, self.beta)

        log_alpha = proposal_log_target - state.log_target
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
) -> RandomWalkMetropolis:
    return RandomWalkMetropolis(model, beta, rng, proposal_scales)
