import numpy as np
from typing import Protocol, runtime_checkable

Array = np.ndarray


@runtime_checkable
class BasisFunction(Protocol):
    name: str
    n_parameters_per_basis: int

    def evaluate(self, x: Array, params: Array) -> Array: ...


class GaussianBasis:
    name = "Gaussian"
    n_parameters_per_basis = 2  # mu, b (amplitude is handled by model)

    def evaluate(self, x: Array, params: Array) -> Array:
        """
        Evaluate K Gaussian basis components at n points.
        params: shape (2, K) -> (mu, b)
        x: shape (n,)
        Returns: shape (K, n)
        """
        mu = params[0]
        b = params[1]
        # Broadcasting: b[:, None] * (x[None, :] - mu[:, None])**2
        return np.exp(-0.5 * b[:, None] * (x[None, :] - mu[:, None]) ** 2)


class LorentzianBasis:
    name = "Lorentzian"
    n_parameters_per_basis = 2  # mu, gamma

    def evaluate(self, x: Array, params: Array) -> Array:
        """
        Evaluate K Lorentzian basis components at n points.
        params: shape (2, K) -> (mu, gamma)
        x: shape (n,)
        Returns: shape (K, n)
        """
        mu = params[0]
        gamma = params[1]
        return 1.0 / (1.0 + ((x[None, :] - mu[:, None]) / gamma[:, None]) ** 2)
