import numpy as np
from .prior import SyntheticPrior

from scipy.special import gamma

Array = np.ndarray


def lgamma(zs: Array | int | float) -> Array:
    return np.log(np.abs(gamma(zs)))


def logsumexp(values: Array) -> float:
    """Stable log(sum(exp(values)))."""
    values = np.asarray(values, dtype=float)
    m = np.max(values)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(values - m))))


def logmeanexp(values: Array) -> float:
    """Stable log(mean(exp(values)))."""
    values = np.asarray(values, dtype=float)
    return logsumexp(values) - np.log(values.size)


def log_gamma_pdf_positive(x: Array, shape: int | float, rate: float) -> Array:
    """Log Gamma(shape, rate) density, x>0."""
    out = shape * np.log(rate) - lgamma(shape) + (shape - 1.0) * np.log(x) - rate * x
    return np.where(x <= 0, -np.inf, out)


def log_normal_pdf(x: Array, mean: int | float, precision: float) -> Array:
    """Log N(mean, precision^{-1}) density."""
    return 0.5 * np.log(precision / (2.0 * np.pi)) - 0.5 * precision * (x - mean) ** 2


def log_prior_theta(theta: Array, K: int, prior: SyntheticPrior) -> float:
    """
    Sum log priors for {a_k, mu_k, b_k}.

    This is exactly the factorized prior form implied by Eqs. (17)-(19).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.size != 3 * K:
        raise ValueError("theta must contain 3*K entries")

    a = theta[:K]
    mu = theta[K : 2 * K]
    b = theta[2 * K :]

    value = 0.0
    value += np.sum(log_gamma_pdf_positive(a, prior.eta_a, prior.lambda_a))
    value += np.sum(log_normal_pdf(mu, prior.nu0, prior.xi0))
    value += np.sum(log_gamma_pdf_positive(b, prior.eta_b, prior.lambda_b))
    return float(value)
