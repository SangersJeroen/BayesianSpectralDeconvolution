"""selection.py — Model-size selection driver."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .evidence import EvidenceEstimate, estimate_evidence
from .model import JaxModel
from .tempering import ExchangeResult, JaxParallelTempering, PTConfig


@dataclass
class JaxModelRun:
    K: int
    model: JaxModel
    exchange_result: ExchangeResult
    evidence: EvidenceEstimate
    seconds: float

    @property
    def stochastic_complexity(self) -> float:
        return -self.evidence.log_z


def select_model_size(
    model_factory: Callable[[int], JaxModel],
    K_values: Sequence[int],
    betas,
    burn_in: int = 2_000,
    samples: int = 2_000,
    seed: int = 1234,
    config: Optional[PTConfig] = None,
    verbose: bool = True,
) -> list[JaxModelRun]:
    """Run batched HMC parallel tempering for each K and estimate log Z."""
    master_rng = np.random.default_rng(seed)
    runs = []
    for K in K_values:
        child_seed = int(master_rng.integers(0, 2**31 - 1))
        model = model_factory(K)
        sampler = JaxParallelTempering(model, betas, config)

        t0 = time.perf_counter()
        result = sampler.run(burn_in=burn_in, samples=samples, seed=child_seed)
        elapsed = time.perf_counter() - t0

        evidence = estimate_evidence(result)
        runs.append(JaxModelRun(K, model, result, evidence, elapsed))

        if verbose:
            print(
                f"K={K:2d} | stochastic complexity -log Z = {runs[-1].stochastic_complexity: .3f} "
                f"| beta=1 HMC acc={result.within_acceptance[-1]:.3f} "
                f"| mean swap acc={np.mean(result.exchange_acceptance):.3f} "
                f"| {elapsed:.1f}s"
            )
    return runs
