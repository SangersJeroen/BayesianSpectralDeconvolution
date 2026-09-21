import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol, Callable
from ..models import SpectralModel

Array = np.ndarray


@dataclass
class MetropolisState:
    theta: Array
    log_target: float
    energy: float
    accepted: int = 0
    attempted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempted if self.attempted else np.nan


class MCMCKernel(Protocol):
    def step(
        self, state: MetropolisState, is_warmup: bool = False
    ) -> MetropolisState: ...


class RandomWalkMetropolis:
    """
    Simple Gaussian random-walk Metropolis sampler.
    """

    def __init__(
        self,
        model: SpectralModel,
        beta: float,
        rng: np.random.Generator,
        proposal_scales: Array,
        use_log_jacobian: bool = True,
    ):
        self.model = model
        self.beta = float(beta)
        self.rng = rng
        self.proposal_scales = np.asarray(proposal_scales, dtype=float)

    def _to_z(self, theta: Array) -> Array:
        to_z = getattr(self.model.parameterization, "to_z", None)
        if callable(to_z):
            return to_z(theta)
        return theta.copy()

    def _from_z(self, z: Array) -> Array:
        from_z = getattr(self.model.parameterization, "from_z", None)
        if callable(from_z):
            return from_z(z)
        return z.copy()

    def _log_jacobian(self, theta: Array, z: Optional[Array] = None) -> float:
        if not self.use_log_jacobian:
            return 0.0
        param = self.model.parameterization
        for method_name in ("log_jacobian", "log_det_jacobian"):
            method = getattr(param, method_name, None)
            if callable(method):
                import inspect

                sig = inspect.signature(method)
                num_params = len(sig.parameters)
                if num_params >= 2:
                    return float(method(theta, z))
                elif num_params == 1:
                    first_param = list(sig.parameters.keys())[0].lower()
                    if first_param in ("z", "z_vec", "unconstrained") and z is not None:
                        return float(method(z))
                    return float(method(theta))
                else:
                    return float(method())
        return 0.0

    def step(self, state: MetropolisState, is_warmup: bool = False) -> MetropolisState:
        to_z, from_z = (
            self.model.parameterization.to_z,
            self.model.parameterization.from_z,
        )
        proposal = from_z(
            to_z(state.theta) + self.rng.normal(0.0, self.proposal_scales)
        )
        proposal_log_target = self.model.log_tempered_target(proposal, self.beta)

            log_alpha = (proposal_log_target - state.log_target) + (
                log_jac_prop - log_jac_old
            )
            accept = np.isfinite(log_alpha) and (
                np.log(self.rng.random()) < min(0.0, float(log_alpha))
            )

        state.attempted += 1
        if accept:
            state.theta = proposal
            state.log_target = proposal_log_target
            state.energy = self.model.energy(proposal)
            state.accepted += 1
        return state


def metropolis_kernel_factory(
    model: SpectralModel,
    beta: float,
    rng: np.random.Generator,
    proposal_scales: Array,
    use_log_jacobian: bool = True,
) -> RandomWalkMetropolis:
    return RandomWalkMetropolis(
        model, beta, rng, proposal_scales, use_log_jacobian=use_log_jacobian
    )
