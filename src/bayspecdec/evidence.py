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
    Estimate log Z(1) using the paper's temperature-ratio approach.

    General formulation:

        Z(beta) = integral p(y | theta)^beta p(theta) dtheta

    and

        Z(beta_{l+1}) / Z(beta_l)
          = E_{q_beta_l}
              [ exp(delta_beta * log p(y | theta)) ]

    Therefore:

        log Z(1)
          = sum_l logmeanexp(
                delta_beta * log_likelihood_samples_l
            )

    This formulation is independent of the observation model.
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
    ratio_standard_errors = []

    for l_index in range(result.beta.size - 1):
        delta_beta = result.beta[l_index + 1] - result.beta[l_index]

        log_likelihoods = np.asarray(
            result.log_likelihood_trace_by_temperature[l_index],
            dtype=float,
        )

        # Each Monte Carlo sample contributes:
        #
        #   exp(delta_beta * log L(theta))
        #
        # Compute in log space to avoid numerical underflow/overflow.
        log_weights = delta_beta * log_likelihoods

        # log E[w]
        log_ratio = logmeanexp(log_weights)
        log_ratios.append(log_ratio)

        # ------------------------------------------------------------
        # Estimate the relative standard error of the ratio.
        #
        # For iid samples:
        #
        #   Var(mean(w)) = Var(w) / M
        #
        # We calculate this in log space as much as possible.
        # ------------------------------------------------------------

        m = log_weights.size

        if m <= 1:
            ratio_standard_errors.append(np.nan)
            continue

        log_mean_w = logmeanexp(log_weights)

        # log(sum(w^2) / M)
        log_mean_w2 = logmeanexp(2.0 * log_weights)

        # Var(w) / mean(w)^2
        #
        # exp(log_mean_w2 - 2 log_mean_w) - 1
        log_cv2 = log_mean_w2 - 2.0 * log_mean_w

        # Numerical roundoff can make this very slightly negative.
        cv2 = max(np.expm1(log_cv2), 0.0)

        # Relative standard error of the sample mean:
        #
        #   SE(mean(w)) / mean(w)
        #       = sqrt(CV^2 / M)
        relative_se = np.sqrt(cv2 / m)

        ratio_standard_errors.append(relative_se)

    log_ratios = np.asarray(log_ratios)
    ratio_standard_errors = np.asarray(ratio_standard_errors)

    return EvidenceEstimate(
        log_z=float(np.sum(log_ratios)),
        log_ratios=log_ratios,
        ratio_standard_errors=ratio_standard_errors,
    )
