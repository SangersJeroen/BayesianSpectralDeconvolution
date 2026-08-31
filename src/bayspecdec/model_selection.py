import numpy as np
from dataclasses import dataclass

from .parallel_tempering import ExchangeResult, ExchangeMonteCarlo
from .evidence import EvidenceEstimate, estimate_log_evidence_from_exchange
from .bayesian_rbf import BayesianRBFProblem

from .functions import beta_schedule
from .prior import SyntheticPrior

Array = np.ndarray


@dataclass
class ModelRun:
    K: int
    problem: BayesianRBFProblem
    exchange_result: ExchangeResult
    evidence: EvidenceEstimate

    @property
    def stochastic_complexity(self) -> float:
        return -self.evidence.log_z

    @property
    def posterior_mode_sample(self) -> Array:
        """
        Approximate MAP using the beta=1 samples.

        The paper reports estimating theta-hat that maximizes posterior
        probability after selecting K. This implementation approximates that
        by taking the sampled beta=1 state with largest log posterior.
        """
        samples = self.exchange_result.samples_by_temperature[-1]
        scores = np.asarray(
            [self.problem.log_target(theta, beta=1.0) for theta in samples]
        )
        return samples[np.argmax(scores)]


def run_model_selection(
    x: Array,
    y: Array,
    sigma2: float,
    K_values: list[int],
    prior: SyntheticPrior,
    L: int = 24,
    burn_in: int = 2_000,
    expectation_steps: int = 2_000,
    seed: int = 1234,
) -> list[ModelRun]:
    """Run exchange MC independently for each candidate K."""
    runs = []
    master_rng = np.random.default_rng(seed)
    beta = beta_schedule(L)

    for K in K_values:
        # Use a fresh deterministic stream per K for reproducibility.
        child_seed = int(master_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(child_seed)
        problem = BayesianRBFProblem(x=x, y=y, sigma2=sigma2, K=K, prior=prior)
        emc = ExchangeMonteCarlo(problem, beta=beta, rng=rng)
        result = emc.run(
            burn_in=burn_in,
            expectation_steps=expectation_steps,
            swap_every=1,
            record_every=1,
            store_state_trace=False,
        )
        evidence = estimate_log_evidence_from_exchange(problem, result)
        runs.append(ModelRun(K, problem, result, evidence))
        print(
            f"K={K:2d} | stochastic complexity -log Z = {runs[-1].stochastic_complexity: .3f} "
            f"| beta=1 MH acc={result.within_acceptance[-1]:.3f} "
            f"| mean swap acc={np.mean(result.exchange_acceptance):.3f}"
        )

    return runs
