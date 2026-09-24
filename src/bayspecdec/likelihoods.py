import numpy as np
from typing import Protocol, runtime_checkable, Optional
from scipy.special import gammaln
from typing import Optional

Array = np.ndarray


@runtime_checkable
class Likelihood(Protocol):
    def log_prob(
        self, y: Array, prediction: Array, context: Optional[dict] = None
    ) -> float: ...

    def energy(self, y: Array, prediction: Array) -> float: ...


class GaussianNoise:
    """
    Gaussian noise observation model.
    y_i = f(x_i) + epsilon_i
    epsilon_i ~ N(0, sigma2)
    """

    def __init__(self, sigma2: float):
        self.sigma2: float = float(sigma2)

    @property
    def pref(self) -> float:
        return 1 / np.sqrt(2 * np.pi * self.sigma2)

    def log_prob(
        self, y: Array, prediction: Array, context: Optional[dict] = None
    ) -> float:
        """Fully normalized log-likelihood."""
        residual: Array = y - prediction
        lprob: float = np.log(self.pref) * (-1 / (2 * self.sigma2) * residual**2).sum()
        return lprob

    def energy(self, y: Array, prediction: Array) -> float:
        return -self.log_prob(y, prediction)


class PoissonNoise:
    """
    Poisson observation model.

    y_i ~ Poisson(lambda_i)
    where lambda_i = prediction_i.

    Unlike the Gaussian noise model, there is no separate sigma2
    parameter: the variance of y_i is equal to lambda_i.
    """

    def log_prob(
        self,
        y: Array,
        prediction: Array,
        context: Optional[dict] = None,
    ) -> float:
        """
        Fully normalized Poisson log-likelihood.

        log p(y | lambda)
            = sum_i [
                y_i * log(lambda_i)
                - lambda_i
                - log(y_i!)
            ]
        """

        y = np.asarray(y)
        prediction = np.asarray(prediction)

        if np.any(y <= 0) or np.any(prediction <= 0):
            raise ValueError(
                "Poisson observations and predictions must be non-negative."
            )

        if not isinstance(y.dtype, np.int_):
            raise ValueError("Poisson observations must be integers.")

        log_likelihood = np.sum(y * np.log(prediction) - prediction - gammaln(y + 1.0))

        return float(log_likelihood)

    def energy(
        self,
        y: Array,
        prediction: Array,
    ) -> float:
        """
        Normalized negative average log-likelihood.

        This is the Poisson analogue of an energy/loss:

            E(theta) = -(1/n) log p(y | theta)

        so that

            log p(y | theta) = -n * E(theta).

        This definition is especially convenient for a generic
        Bayesian/tempered inference framework.
        """
        n = y.size
        return -self.log_prob(y, prediction) / n
