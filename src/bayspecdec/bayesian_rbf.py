import numpy as np
from dataclasses import dataclass

from .gaussian_rbf import mean_squared_error
from .likelihood import log_prior_theta

Array = np.ndarray

@dataclass
class BayesianRBFProblem:
    x: Array
    y: Array
    sigma2: float
    K: int
    prior: SyntheticPrior = SyntheticPrior()

    def energy(self, theta: Array) -> float:
        return mean_squared_error(self.x, self.y, theta)

    def log_likelihood(self, theta: Array) -> float:
        """
        Log likelihood up to the Gaussian normalizing constant.

        For a fixed sigma^2, the paper's q(theta; beta) only needs
            -(n/sigma^2) beta E(theta) + log prior.
        """
        n = self.x.size
        return -(n / self.sigma2) * self.energy(theta)

    def log_target(self, theta: Array, beta: float) -> float:
        """Log of q(theta; beta), up to a beta-dependent normalization constant."""
        lp = log_prior_theta(theta, self.K, self.prior)
        if not np.isfinite(lp):
            return -np.inf
        return beta * self.log_likelihood(theta) + lp
