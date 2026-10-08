"""evidence.py — Marginal-likelihood estimate from the temperature ladder of a parallel-tempering run."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .tempering import ExchangeResult

Array = np.ndarray


def logmeanexp(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    return float(logsumexp(values) - np.log(values.size))


@dataclass
class EvidenceEstimate:
    log_z: float
    log_ratios: Array
    ratio_standard_errors: Array


def estimate_evidence(
    result: ExchangeResult, verbose: bool = False
) -> EvidenceEstimate:
    """
    Estimate log Z(1) from the temperature ratios (Nagata et al. 2012).

    With ``Z(beta) = integral p(y | theta)^beta p(theta) dtheta``,

        Z(beta_{l+1}) / Z(beta_l) = E_{q_beta_l}[exp(delta_beta * log p(y | theta))],

    so ``log Z(1) = sum_l logmeanexp(delta_beta_l * log_likelihood_samples_l)`` (with ``Z(0) = 1``
    for a normalised prior). This only needs the log-likelihood traces, so it holds for any noise model.
    Each ratio's relative standard error assumes independent samples.
    """
    if result.beta.size < 2:
        raise ValueError(
            "At least two beta values are required for evidence estimation."
        )
    if len(result.log_likelihood_trace_by_temperature) != result.beta.size:
        raise ValueError(
            "Number of likelihood traces must match number of beta values."
        )

    log_ratios = []
    standard_errors = []

    for l in range(result.beta.size - 1):
        delta_beta = result.beta[l + 1] - result.beta[l]
        log_likelihoods = np.asarray(
            result.log_likelihood_trace_by_temperature[l], dtype=float
        )

        if log_likelihoods.size == 0:
            raise ValueError(f"No samples available for beta index {l}.")
        if np.any(np.isposinf(log_likelihoods)) or np.any(np.isnan(log_likelihoods)):
            warnings.warn(
                f"Infinite or NaN log-likelihood values at beta index {l}.",
                RuntimeWarning,
            )

        log_w = delta_beta * log_likelihoods  # log of the importance weights
        log_ratio = logmeanexp(log_w)
        log_ratios.append(log_ratio)

        m = log_w.size
        if m > 1:
            # relative SE of the mean: sqrt(CV^2 / m) with CV^2 = E[w^2] / E[w]^2 - 1
            cv2 = max(np.expm1(logmeanexp(2.0 * log_w) - 2.0 * log_ratio), 0.0)
            standard_errors.append(np.sqrt(cv2 / m))
        else:
            standard_errors.append(np.nan)

        if verbose:
            weight_ess = np.exp(2 * logsumexp(log_w) - logsumexp(2 * log_w))
            print(
                f"{l:2d} beta={result.beta[l]:.4f}->{result.beta[l + 1]:.4f} "
                f"Δβ={delta_beta:.4f} log-ratio={log_ratio:.4f} weight-ESS={weight_ess:.1f}/{m}"
            )

    log_z = float(np.sum(log_ratios))
    if verbose:
        print(f"Calculated log(Z)={log_z:.2f}")

    return EvidenceEstimate(log_z, np.asarray(log_ratios), np.asarray(standard_errors))
