import numpy as np
from typing import Protocol, runtime_checkable

Array = np.ndarray


@runtime_checkable
class Parameterization(Protocol):
    ndim: int
    K: int

    def pack(self, structured) -> Array: ...
    def unpack(self, theta: Array): ...
    def validate(self, theta: Array) -> None: ...
    def to_z(self, theta: Array) -> Array: ...
    def from_z(self, z: Array) -> Array: ...



class DefaultParameterization:
    """
    Standard parameterization from the paper.
    theta = [a_1...a_K, mu_1...mu_K, b_1...b_K]
    """

    def __init__(self, K: int):
        self.K = K
        self.ndim = 3 * K

    def pack(self, structured: tuple[Array, Array, Array]) -> Array:
        a, mu, b = structured
        return np.concatenate([a, mu, b])

    def unpack(self, theta: Array) -> tuple[Array, Array, Array]:
        theta = np.asarray(theta, dtype=float)
        return (
            theta[..., : self.K],
            theta[..., self.K : 2 * self.K],
            theta[..., 2 * self.K :],
        )

    def validate(self, theta: Array) -> None:
        if theta.size != self.ndim:
            raise ValueError(f"Expected {self.ndim} parameters, got {theta.size}")

    def to_z(self, theta: Array) -> Array:
        return np.asarray(theta, dtype=float).copy()

    def from_z(self, z: Array) -> Array:
        return np.asarray(z, dtype=float).copy()

    def log_jacobian(self, theta: Array) -> float:
        return 0.0
