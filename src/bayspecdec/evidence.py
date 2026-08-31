import numpy as np
from dataclasses import dataclass

from .bayesian_rbf import BayesianRBFProblem
from .parallel_tempering import ExchangeResult
from .likelihood import logmeanexp

Array = np.ndarray


@dataclass
class EvidenceEstimate:
    log_z: float
    log_ratios: Array
    ratio_standard_errors: Array


def estimate_log_evidence_from_exchange(
    problem: BayesianRBFProblem,
    result: ExchangeResult,
) -> EvidenceEstimate:
    """
    Estimate log Z(1) using Eq. (10).

    For each adjacent pair:

        Z(beta_{l+1}) / Z(beta_l)
          = E_{q_beta_l}[ exp(-n/sigma^2 * (beta_{l+1}-beta_l) E(theta)) ]

    We use the samples produced at beta_l and compute the expectation by
    ordinary Monte Carlo averaging.
    """
    n = problem.x.size
    sigma2 = problem.sigma2
    log_ratios = []
    ratio_se = []

    for l in range(result.beta.size - 1):
        delta_beta = result.beta[l + 1] - result.beta[l]
        energies = result.energy_trace_by_temperature[l]
        if energies.size == 0:
            raise ValueError("No samples available for evidence estimation")

        log_weights = -(n / sigma2) * delta_beta * energies
        log_ratio = logmeanexp(log_weights)
        log_ratios.append(log_ratio)

        # Monte Carlo standard error on the ratio itself.
        weights = np.exp(log_weights - np.max(log_weights))
        weights *= np.exp(np.max(log_weights))
        ratio = np.mean(weights)
        if energies.size > 1:
            se = np.std(weights, ddof=1) / np.sqrt(energies.size)
        else:
            se = np.nan
        ratio_se.append(se / max(ratio, 1e-300))

    log_ratios = np.asarray(log_ratios)
    return EvidenceEstimate(
        log_z=float(np.sum(log_ratios)),
        log_ratios=log_ratios,
        ratio_standard_errors=np.asarray(ratio_se),
    )
