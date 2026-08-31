from dataclasses import dataclass
import numpy as np

from typing import Optional

from .bayesian_rbf import BayesianRBFProblem

Array = np.ndarray

@dataclass
class MetropolisState:
    theta: Array
    log_target: float
    accepted: int = 0
    attempted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempted if self.attempted else np.nan


class RandomWalkMetropolis:
    """
    Simple Gaussian random-walk Metropolis sampler.

    Proposal:
        theta' = theta + Normal(0, proposal_std^2)

    The proposal is symmetric, so the proposal densities cancel and
    acceptance is min(1, exp(log_target' - log_target)).
    """

    def __init__(
        self,
        problem: BayesianRBFProblem,
        beta: float,
        rng: np.random.Generator,
        proposal_scales: Optional[Array] = None,
    ) -> None:
        self.problem = problem
        self.beta = float(beta)
        self.rng = rng
        if proposal_scales is None:
            # Crude defaults that work for the educational demo. In a serious
            # implementation, tune these using pilot acceptance rates.
            K = problem.K
            proposal_scales = np.concatenate([
                np.full(K, 0.03),   # strengths
                np.full(K, 0.02),   # centers
                np.full(K, 2.0),    # b parameters
            ])
        self.proposal_scales = np.asarray(proposal_scales, dtype=float)

    def step(self, state: MetropolisState) -> MetropolisState:
        proposal = state.theta + self.rng.normal(0.0, self.proposal_scales)
        proposal_log_target = self.problem.log_target(proposal, self.beta)

        log_alpha = proposal_log_target - state.log_target
        accept = np.log(self.rng.random()) < min(0.0, log_alpha)

        state.attempted += 1
        if accept:
            state.theta = proposal
            state.log_target = proposal_log_target
            state.accepted += 1
        return state
