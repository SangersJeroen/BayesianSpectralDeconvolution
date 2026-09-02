import numpy as np
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

    def energy(self, theta: Array) -> float:
        pred = self.predict(theta)
        return self.likelihood.energy(self.y, pred)

    def log_likelihood(self, theta: Array) -> float:
        pred = self.predict(theta)
        return self.likelihood.log_prob(self.y, pred)

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

        # To strictly match the paper's q_beta(theta) which omits the Gaussian normalizer
        # We calculate: -(n / sigma2) * beta * E(theta)
        # Note: self.likelihood is expected to have a sigma2 attribute for GaussianNoise,
        # but in a truly generic setup we might just say:
        # beta * log_likelihood, unless we want to avoid the log_likelihood normalization constant.
        # The paper explicitly states -(n / sigma2) * beta * E(theta).
        # We can reconstruct it if likelihood has sigma2:
        if hasattr(self.likelihood, "sigma2"):
            sigma2 = self.likelihood.sigma2
            ll_part = -(self.n / sigma2) * self.energy(theta)
        else:
            # Fallback for arbitrary likelihood: use fully normalized log_prob
            # But wait, tempering usually tempers the whole likelihood.
            ll_part = self.log_likelihood(theta)

        return beta * ll_part + lp
