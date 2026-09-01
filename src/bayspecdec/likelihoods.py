import numpy as np
from typing import Protocol, runtime_checkable, Optional

Array = np.ndarray

@runtime_checkable
class Likelihood(Protocol):
    def log_prob(self, y: Array, prediction: Array, context: Optional[dict] = None) -> float:
        ...
    
    def energy(self, y: Array, prediction: Array) -> float:
        ...

class GaussianNoise:
    """
    Gaussian noise observation model.
    y_i = f(x_i) + epsilon_i
    epsilon_i ~ N(0, sigma2)
    """
    def __init__(self, sigma2: float):
        self.sigma2 = float(sigma2)
        
    def log_prob(self, y: Array, prediction: Array, context: Optional[dict] = None) -> float:
        """Fully normalized log-likelihood."""
        n = y.size
        # log N(y; f(x), sigma2*I) = -n/2 log(2*pi*sigma2) - 1/(2*sigma2) sum (y - f(x))^2
        residual = y - prediction
        return -0.5 * n * np.log(2.0 * np.pi * self.sigma2) - (0.5 / self.sigma2) * np.sum(residual ** 2)

    def energy(self, y: Array, prediction: Array) -> float:
        """Paper's E(theta): 1/(2n) * sum squared residuals."""
        n = y.size
        residual = y - prediction
        return float(0.5 * np.mean(residual ** 2))
