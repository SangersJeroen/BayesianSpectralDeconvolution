import numpy as np
from typing import Optional
from .basis import BasisFunction
from .priors import Prior
from .likelihoods import Likelihood
from .parameters import Parameterization

Array = np.ndarray


class SpectralModel:
    """
    Combines a forward model (basis), likelihood, prior, and data.
    """

    def __init__(
        self,
        x: Array,
        y: Array,
        basis: BasisFunction,
        prior: Prior,
        likelihood: Likelihood,
        parameterization: Parameterization,
    ):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.basis = basis
        self.prior = prior
        self.likelihood = likelihood
        self.parameterization = parameterization
        self.n = self.x.size

        n_noise = int(getattr(parameterization, "n_noise", 0))
        n_lik = int(getattr(likelihood, "n_noise_params", 0))
        if n_noise != n_lik:
            raise ValueError(
                f"The likelihood reads {n_lik} noise hyperparameter(s) but the parameterization "
                f"has {n_noise}. For a sampled sigma^2 use GaussianNoise(sigma2=None) with a "
                "parameterization built with n_noise=1 (and a prior with a noise entry)."
            )
        self.n_noise = n_noise

    def predict(self, theta: Array) -> Array:
        # Expected structure: a, basis_params...
        # For simplicity, assuming parameterization unpack returns (amplitudes, param1, param2, ...)
        # In DefaultParameterization it returns (a, mu, b)
        unpacked = self.parameterization.unpack(theta)
        a = unpacked[0]
        basis_params = unpacked[1:]  # e.g. (mu, b)

        # We need to pass the basis parameters correctly.
        # Since basis_params is a tuple of arrays, we can stack them into shape (n_params, K)
        basis_params_array = np.stack(basis_params)

        basis_matrix = self.basis.evaluate(self.x, basis_params_array)  # shape (K, n)
        return a @ basis_matrix

    def noise_context(self, theta: Array) -> Optional[dict]:
        """Noise hyperparameters in ``theta`` for the likelihood (``None`` if there are none)."""
        if self.n_noise == 0:
            return None
        return {"noise": self.parameterization.noise(theta)}

    def energy(self, theta: Array) -> float:
        pred = self.predict(theta)
        return self.likelihood.energy(self.y, pred, self.noise_context(theta))

    def log_likelihood(self, theta: Array) -> float:
        pred = self.predict(theta)
        return self.likelihood.log_prob(self.y, pred, self.noise_context(theta))

    def log_prior(self, theta: Array) -> float:
        return self.prior.log_prob(theta, self.parameterization)

    def log_posterior(self, theta: Array) -> float:
        return self.log_tempered_target(theta, 1.0)

    def log_tempered_target(self, theta: Array, beta: float) -> float:
        """
        Log of q_beta(theta) up to beta-dependent normalization.
        Paper: - (n/sigma^2) * beta * E(theta) + log prior
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        ll_part = self.log_likelihood(theta)

        return beta * ll_part + lp

    def log_target_z(self, z: Array, beta: float) -> float:
        """
        Log density of the tempered target in the unconstrained coordinates ``z``:
        ``log q_beta(theta(z)) + log |d theta / d z|``. This is what gradient-based samplers use.
        """
        theta = self.parameterization.from_z(z)
        lt = self.log_tempered_target(theta, beta)
        if not np.isfinite(lt):
            return -np.inf
        return lt + self.parameterization.log_jacobian(theta)
