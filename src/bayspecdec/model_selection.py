import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence

from .tempering import ExchangeResult, ParallelTempering
from .evidence import EvidenceEstimate, estimate_evidence
from .models import SpectralModel

Array = np.ndarray


@dataclass
class ModelRun:
    K: int
    model: SpectralModel
    exchange_result: ExchangeResult
    evidence: EvidenceEstimate

    @property
    def stochastic_complexity(self) -> float:
        return -self.evidence.log_z

    @property
    def posterior_mode_sample(self) -> Array:
        """Approximate MAP using the beta=1 samples."""
        samples = self.exchange_result.samples_by_temperature[-1]
        scores = np.asarray([self.model.log_posterior(theta) for theta in samples])
        return samples[np.argmax(scores)]


def select_model_size(
    model_factory: Callable[[int], SpectralModel],
    K_values: Sequence[int],
    sampler_factory: Callable[[SpectralModel, np.random.Generator], ParallelTempering],
    burn_in: int = 2_000,
    samples: int = 2_000,
    seed: int = 1234,
    swap_every: int = 1,
    store_state_trace: bool = False,
) -> list[ModelRun]:
    """Run parallel tempering independently for each candidate K and estimate evidence."""
    runs = []
    master_rng = np.random.default_rng(seed)

    for K in K_values:
        child_seed = int(master_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(child_seed)

        model = model_factory(K)
        sampler = sampler_factory(model, rng)

        result = sampler.run(
            burn_in=burn_in,
            samples=samples,
            swap_every=swap_every,
            record_every=1,
            store_state_trace=store_state_trace,
        )

        evidence = estimate_evidence(model, result)
        runs.append(ModelRun(K, model, result, evidence))

        print(
            f"K={K:2d} | stochastic complexity -log Z = {runs[-1].stochastic_complexity: .3f} "
            f"| beta=1 MH acc={result.within_acceptance[-1]:.3f} "
            f"| mean swap acc={np.mean(result.exchange_acceptance):.3f}"
        )

    return runs
