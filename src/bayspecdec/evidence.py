import numpy as np
from dataclasses import dataclass
from .models import SpectralModel
from .tempering import ExchangeResult

Array = np.ndarray

def logsumexp(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    m = np.max(values)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(values - m))))

def logmeanexp(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    return logsumexp(values) - np.log(values.size)

@dataclass
class EvidenceEstimate:
    log_z: float
    log_ratios: Array
    ratio_standard_errors: Array

def estimate_evidence(
    model: SpectralModel,
    result: ExchangeResult,
) -> EvidenceEstimate:
    """
    Estimate log Z(1) using Eq. (10).
    """
    n = model.n
    
    # Needs sigma2 for the paper formula
    if hasattr(model.likelihood, 'sigma2'):
        sigma2 = model.likelihood.sigma2
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

            # Standard error estimation
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
    else:
        # Generic case: uses full log_likelihood differences
        # But wait, we'd need log_likelihood evaluated for all samples. 
        # For now, just raise an error or assume we only use models with energies/sigma2.
        raise NotImplementedError("Evidence estimation currently requires a likelihood with sigma2 and energies.")
