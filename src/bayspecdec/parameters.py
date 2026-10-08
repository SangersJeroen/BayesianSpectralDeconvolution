import numpy as np
from scipy.special import expit
from typing import Optional, Protocol, Sequence, runtime_checkable

Array = np.ndarray

_EPS = 1e-12


@runtime_checkable
class Parameterization(Protocol):
    """
    Maps between the flat parameter vector ``theta`` and structured blocks, and between
    ``theta`` and the unconstrained coordinates ``z`` the gradient-based samplers move in.

    ``log_jacobian(theta)`` is ``log |det d theta / d z|``; it turns a density in theta-space
    into the density in z-space that the samplers actually target.
    """

    ndim: int
    K: int

    def pack(self, structured) -> Array: ...
    def unpack(self, theta: Array): ...
    def validate(self, theta: Array) -> None: ...
    def to_z(self, theta: Array) -> Array: ...
    def from_z(self, z: Array) -> Array: ...
    def log_jacobian(self, theta: Array) -> float: ...


class BlockParameterization:
    """
    ``theta = [a_1..a_K, p1_1..p1_K, ..., pm_1..pm_K, s_1..s_n]``.

    One block of ``K`` amplitudes, one block of ``K`` values per basis parameter, then
    ``n_noise`` noise hyperparameters (e.g. ``log sigma^2``) that belong to the likelihood and
    are shared by all components. ``unpack`` returns ``(a, p1, ..., pm)`` and ignores the noise
    block, which ``noise`` returns. This is the layout of ``jax_backend.make_spectral_model``.

    The ``to_z`` / ``from_z`` transform is element-wise and follows each block's bounds:

    * both bounds finite -> logistic map onto ``(lower, upper)``
    * only the lower bound finite -> ``lower + exp(z)``
    * no finite bound -> identity

    so gradient-based samplers never leave the prior support.

    Parameters
    ----------
    block_bounds:
        ``(lower, upper)`` for the amplitude block followed by each basis-parameter block.
    noise_bounds:
        ``(lower, upper)`` for each noise hyperparameter.
    """

    def __init__(
        self,
        K: int,
        block_bounds: Sequence[tuple[float, float]],
        noise_bounds: Sequence[tuple[float, float]] = (),
    ):
        if K < 1:
            raise ValueError("K must be >= 1")
        if len(block_bounds) < 1:
            raise ValueError("Need bounds for at least the amplitude block")
        self.K = int(K)
        self.n_blocks = len(block_bounds)
        self.n_noise = len(noise_bounds)
        self.ndim = self.n_blocks * self.K + self.n_noise

        lower = np.concatenate(
            [np.full(self.K, lo, dtype=float) for lo, _ in block_bounds]
            + [np.full(1, lo, dtype=float) for lo, _ in noise_bounds]
        )
        upper = np.concatenate(
            [np.full(self.K, hi, dtype=float) for _, hi in block_bounds]
            + [np.full(1, hi, dtype=float) for _, hi in noise_bounds]
        )
        if np.any(np.isfinite(upper) & ~np.isfinite(lower)):
            raise ValueError("Upper-only bounds are not supported")
        if np.any(upper <= lower):
            raise ValueError("upper must exceed lower")

        self.lower, self.upper = lower, upper
        self._both = np.isfinite(lower) & np.isfinite(upper)
        self._low = np.isfinite(lower) & ~np.isfinite(upper)
        self._lo = np.where(np.isfinite(lower), lower, 0.0)
        self._width = np.where(self._both, upper - lower, 1.0)

    # -- layout -----------------------------------------------------------------

    def pack(self, structured) -> Array:
        return np.concatenate([np.asarray(b, dtype=float) for b in structured])

    def unpack(self, theta: Array) -> tuple[Array, ...]:
        theta = np.asarray(theta, dtype=float)
        K = self.K
        return tuple(theta[..., i * K : (i + 1) * K] for i in range(self.n_blocks))

    def noise(self, theta: Array) -> Array:
        """The trailing ``n_noise`` hyperparameters (shape ``(..., n_noise)``)."""
        return np.asarray(theta, dtype=float)[..., self.n_blocks * self.K :]

    def validate(self, theta: Array) -> None:
        if np.size(theta) != self.ndim:
            raise ValueError(f"Expected {self.ndim} parameters, got {np.size(theta)}")

    # -- unconstrained coordinates ----------------------------------------------

    def from_z(self, z: Array) -> Array:
        z = np.asarray(z, dtype=float)
        bounded = self._lo + self._width * expit(z)
        with np.errstate(over="ignore"):
            positive = self._lo + np.exp(z)
        return np.where(self._both, bounded, np.where(self._low, positive, z))

    def to_z(self, theta: Array) -> Array:
        theta = np.asarray(theta, dtype=float)
        u = np.clip((theta - self._lo) / self._width, _EPS, 1.0 - _EPS)
        bounded = np.log(u) - np.log1p(-u)
        positive = np.log(np.maximum(theta - self._lo, _EPS))
        return np.where(self._both, bounded, np.where(self._low, positive, theta))

    def log_jacobian(self, theta: Array) -> float:
        theta = np.asarray(theta, dtype=float)
        u = np.clip((theta - self._lo) / self._width, _EPS, 1.0 - _EPS)
        bounded = np.log(self._width) + np.log(u) + np.log1p(-u)
        positive = np.log(np.maximum(theta - self._lo, _EPS))
        return float(np.sum(np.where(self._both, bounded, np.where(self._low, positive, 0.0))))


class DefaultParameterization(BlockParameterization):
    """
    The paper's layout ``theta = [a_1..a_K, mu_1..mu_K, b_1..b_K]`` (+ optional noise block),
    with ``a, b > 0`` (``exp`` map) and ``mu`` unbounded. Pass ``mu_bounds=(lo, hi)`` to
    confine the centres to a box (logistic map).
    """

    def __init__(
        self,
        K: int,
        n_noise: int = 0,
        noise_bounds: Optional[Sequence[tuple[float, float]]] = None,
        mu_bounds: tuple[float, float] = (-np.inf, np.inf),
    ):
        if noise_bounds is None:
            noise_bounds = [(-np.inf, np.inf)] * n_noise
        elif len(noise_bounds) != n_noise and n_noise:
            raise ValueError("noise_bounds must have n_noise entries")
        super().__init__(
            K,
            block_bounds=[(0.0, np.inf), tuple(mu_bounds), (0.0, np.inf)],
            noise_bounds=noise_bounds,
        )
