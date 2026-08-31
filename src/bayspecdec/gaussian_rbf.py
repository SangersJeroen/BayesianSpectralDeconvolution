import numpy as np

Array = np.ndarray

# -----------------------------------------------------------------------------
# Model: sum of Gaussian radial basis functions
# -----------------------------------------------------------------------------

def gaussian_rbf(x: Array, mu: float, b: float) -> Array:
    """
    Gaussian basis used in Eq. (2):

        phi(x) = exp[- b/2 * (x - mu)^2]

    Notice that b is a *precision-like bandwidth parameter*: larger b means
    a narrower Gaussian, because the exponent becomes more negative faster.
    """
    return np.exp(-0.5 * b * (x - mu) ** 2)


def gaussian_rbf_spectrum(x: Array, theta: Array) -> Array:
    """
    Evaluate the model

        f(x; theta) = sum_k a_k phi_k(x)

    theta is stored as [a_1,...,a_K, mu_1,...,mu_K, b_1,...,b_K].
    """
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    K = theta.size // 3
    a = theta[:K]
    mu = theta[K:2 * K]
    b = theta[2 * K:]

    # Shape: (K, n)
    basis = np.exp(-0.5 * b[:, None] * (x[None, :] - mu[:, None]) ** 2)
    return a @ basis


def mean_squared_error(x: Array, y: Array, theta: Array) -> float:
    """Paper's E(theta) in Eq. (3): 1/(2n) * sum squared residuals."""
    residual = y - gaussian_rbf_spectrum(x, theta)
    return float(0.5 * np.mean(residual ** 2))
