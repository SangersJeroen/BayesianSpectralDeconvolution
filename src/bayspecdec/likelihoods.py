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
        lprob: float = (
            np.log(self.pref)
            * residual.size
            * -1
            / (2 * self.sigma2)
            * (residual**2).sum()
        )
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


class PoissonGaussianNoise:
    """
    Poisson-Gaussian observation model.

    Latent count:
        k_i ~ Poisson(prediction_i)

    Readout:
        y_i | k_i ~ N(k_i, sigma2)

    Therefore:
        p(y_i | prediction_i, sigma2)
            = sum_k Poisson(k | prediction_i) N(y_i | k, sigma2)

    The public interface evaluates the likelihood of a complete
    spectrum in one call.
    """

    def __init__(
        self,
        sigma2: float,
        tol: float = 1e-12,
        max_iterations: int = 100_000,
    ):
        if sigma2 <= 0:
            raise ValueError("sigma2 must be strictly positive.")

        if not 0.0 < tol < 1.0:
            raise ValueError("tol must lie in (0, 1).")

        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")

        self.sigma2: float = float(sigma2)
        self.tol: float = float(tol)
        self.max_iterations: int = int(max_iterations)

        self._log_pref: float = -0.5 * np.log(2.0 * np.pi * self.sigma2)

    @property
    def pref(self) -> float:
        return 1.0 / np.sqrt(2.0 * np.pi * self.sigma2)

    def _log_ratio(
        self,
        k: np.typing.NDArray[np.integer],
        log_lambda: Array,
        y: Array,
    ) -> Array:
        """
        log(a_{k+1} / a_k), where

            a_k = Poisson(k | lambda) * N(y | k, sigma2)

        The ratio is strictly decreasing with k.
        """
        k_float = np.asarray(k, dtype=float)

        return log_lambda - np.log(k_float + 1.0) + (y - k_float - 0.5) / self.sigma2

    def _find_modes(
        self,
        y: Array,
        prediction: Array,
    ) -> np.typing.NDArray[np.integer]:
        """
        Find the integer mode of the summand for every spectrum bin.

        Because log(a_{k+1}/a_k) is strictly decreasing in k, the mode
        can be found using bracketing followed by binary search.
        """

        n = y.size
        modes = np.zeros(n, dtype=np.int64)

        positive = prediction > 0.0

        if not np.any(positive):
            return modes

        log_lambda = np.full(n, -np.inf, dtype=float)
        log_lambda[positive] = np.log(prediction[positive])

        # If a_1 <= a_0, then k=0 is the mode.
        ratio0 = self._log_ratio(
            np.zeros(n, dtype=np.int64),
            log_lambda,
            y,
        )

        needs_search = positive & (ratio0 > 0.0)

        if not np.any(needs_search):
            return modes

        # ------------------------------------------------------------
        # Bracket the mode.
        #
        # Find lo, hi such that:
        #   log_ratio(lo) > 0
        #   log_ratio(hi) <= 0
        # ------------------------------------------------------------
        lo = np.zeros(n, dtype=np.int64)
        hi = np.zeros(n, dtype=np.int64)

        hi[needs_search] = 1

        active = needs_search.copy()

        while np.any(active):
            idx = np.flatnonzero(active)

            ratio = self._log_ratio(
                hi[idx],
                log_lambda[idx],
                y[idx],
            )

            still_positive = ratio > 0.0

            hi_values = hi[idx]
            hi_values[still_positive] = 2 * hi_values[still_positive] + 1
            hi[idx] = hi_values

            active[idx[~still_positive]] = False

        # ------------------------------------------------------------
        # Binary search for the first k with log_ratio(k) <= 0.
        # ------------------------------------------------------------
        active = needs_search.copy()

        while np.any(active):
            idx = np.flatnonzero(active)

            unfinished = (hi[idx] - lo[idx]) > 1

            if not np.any(unfinished):
                active[idx] = False
                continue

            jj = idx[unfinished]

            mid = (lo[jj] + hi[jj]) // 2

            ratio = self._log_ratio(
                mid,
                log_lambda[jj],
                y[jj],
            )

            positive_ratio = ratio > 0.0

            lo_jj = lo[jj]
            hi_jj = hi[jj]

            lo_jj[positive_ratio] = mid[positive_ratio]
            hi_jj[~positive_ratio] = mid[~positive_ratio]

            lo[jj] = lo_jj
            hi[jj] = hi_jj

            active[idx] = (hi[idx] - lo[idx]) > 1

        modes[needs_search] = hi[needs_search]

        return modes

    @staticmethod
    def _log_geometric_tail(
        log_a: Array,
        log_q: Array,
    ) -> Array:
        """
        Upper bound

            a * q / (1-q)

        in log space, assuming q < 1.
        """

        result = np.full_like(log_a, np.inf, dtype=float)

        valid = log_q < 0.0

        if np.any(valid):
            lq = log_q[valid]

            # log(1 - exp(lq)), evaluated stably.
            log_one_minus_q = np.log(-np.expm1(lq))

            result[valid] = log_a[valid] + lq - log_one_minus_q

        return result

    def log_prob(
        self,
        y: Array,
        prediction: Array,
        context: Optional[dict] = None,
    ) -> float:
        """
        Fully normalized log-likelihood of the complete spectrum.

        Parameters
        ----------
        y:
            Observed spectrum.

        prediction:
            Poisson mean f(x_i; theta) for each bin.

        context:
            Optional extra information. Currently unused.

        Returns
        -------
        float
            Sum_i log p(y_i | prediction_i, sigma2).
        """

        y = np.asarray(y, dtype=float)
        prediction = np.asarray(prediction, dtype=float)

        if y.shape != prediction.shape:
            raise ValueError("y and prediction must have the same shape.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values.")

        if not np.all(np.isfinite(prediction)):
            raise ValueError("prediction contains non-finite values.")

        if np.any(prediction < 0.0):
            raise ValueError(
                "prediction must be non-negative because it is a Poisson mean."
            )

        # Flatten the spectrum. The original shape is irrelevant for
        # the independent-bin likelihood.
        y = y.ravel()
        prediction = prediction.ravel()

        n = y.size

        positive = prediction > 0.0

        log_lambda = np.full(n, -np.inf, dtype=float)
        log_lambda[positive] = np.log(prediction[positive])

        # ------------------------------------------------------------
        # Find the mode of each Poisson-Gaussian summand.
        # ------------------------------------------------------------
        k_mode = self._find_modes(y, prediction)

        k = k_mode.copy()

        # ------------------------------------------------------------
        # log(a_k) at the mode
        #
        # log a_k =
        #   k log(lambda)
        #   - lambda
        #   - log Gamma(k+1)
        #   + log Gaussian prefactor
        #   - (y-k)^2/(2 sigma2)
        # ------------------------------------------------------------
        log_a = np.empty(n, dtype=float)

        zero_lambda = ~positive

        # If lambda=0, only k=0 contributes.
        log_a[zero_lambda] = self._log_pref - y[zero_lambda] ** 2 / (2.0 * self.sigma2)

        log_a[positive] = (
            k[positive] * log_lambda[positive]
            - prediction[positive]
            - gammaln(k[positive] + 1.0)
            + self._log_pref
            - (y[positive] - k[positive]) ** 2 / (2.0 * self.sigma2)
        )

        # log of the currently retained sum
        log_sum = log_a.copy()

        # ============================================================
        # RIGHT TAIL
        # ============================================================

        active = positive.copy()
        log_tail_tolerance = np.log(self.tol / 2.0)

        iteration = 0

        while np.any(active):
            iteration += 1

            if iteration > self.max_iterations:
                raise RuntimeError(
                    "Maximum right-tail iterations exceeded. "
                    "Increase max_iterations or inspect the input."
                )

            idx = np.flatnonzero(active)

            current_k = k[idx]

            log_q = self._log_ratio(
                current_k,
                log_lambda[idx],
                y[idx],
            )

            # Upper bound on omitted right tail.
            log_tail = self._log_geometric_tail(
                log_a[idx],
                log_q,
            )

            converged = log_tail <= log_sum[idx] + log_tail_tolerance

            active[idx[converged]] = False

            not_converged = idx[~converged]

            if not_converged.size:
                log_q_next = self._log_ratio(
                    k[not_converged],
                    log_lambda[not_converged],
                    y[not_converged],
                )

                k[not_converged] += 1
                log_a[not_converged] += log_q_next

                # Stable:
                # log(exp(a) + exp(b))
                log_sum[not_converged] = np.logaddexp(
                    log_sum[not_converged],
                    log_a[not_converged],
                )

        # ------------------------------------------------------------
        # Reset k/log_a to the mode before expanding left.
        #
        # log_sum intentionally retains all right-side contributions.
        # ------------------------------------------------------------
        k = k_mode.copy()

        log_a = np.empty(n, dtype=float)

        log_a[zero_lambda] = self._log_pref - y[zero_lambda] ** 2 / (2.0 * self.sigma2)

        log_a[positive] = (
            k[positive] * log_lambda[positive]
            - prediction[positive]
            - gammaln(k[positive] + 1.0)
            + self._log_pref
            - (y[positive] - k[positive]) ** 2 / (2.0 * self.sigma2)
        )

        # ============================================================
        # LEFT TAIL
        # ============================================================

        active = positive & (k > 0)

        iteration = 0

        while np.any(active):
            iteration += 1

            if iteration > self.max_iterations:
                raise RuntimeError(
                    "Maximum left-tail iterations exceeded. "
                    "Increase max_iterations or inspect the input."
                )

            idx = np.flatnonzero(active)

            # a_{k-1} / a_k = 1 / r_{k-1}
            log_q = -self._log_ratio(
                k[idx] - 1,
                log_lambda[idx],
                y[idx],
            )

            # Upper bound on omitted left tail.
            log_tail = self._log_geometric_tail(
                log_a[idx],
                log_q,
            )

            converged = log_tail <= log_sum[idx] + log_tail_tolerance

            active[idx[converged]] = False

            not_converged = idx[~converged]

            if not_converged.size:
                log_ratio_previous = self._log_ratio(
                    k[not_converged] - 1,
                    log_lambda[not_converged],
                    y[not_converged],
                )

                k[not_converged] -= 1

                # log a_{k-1} = log a_k - log r_{k-1}
                log_a[not_converged] -= log_ratio_previous

                log_sum[not_converged] = np.logaddexp(
                    log_sum[not_converged],
                    log_a[not_converged],
                )

                # At k=0 there is no remaining left tail.
                reached_zero = k[not_converged] == 0
                active[not_converged[reached_zero]] = False

        # ------------------------------------------------------------
        # log_sum[i] is now the truncated marginal log likelihood
        # for each individual spectrum bin.
        #
        # The full-spectrum log likelihood is the sum because bins
        # are conditionally independent.
        # ------------------------------------------------------------
        return float(np.sum(log_sum))

    def energy(
        self,
        y: Array,
        prediction: Array,
    ) -> float:
        """
        Negative log-likelihood of the complete spectrum.
        """
        return -self.log_prob(y, prediction)
