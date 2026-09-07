import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from ..models import SpectralModel
from .metropolis import MetropolisState, MCMCKernel

Array = np.ndarray

@dataclass
class HMCConfig:
    num_steps: int
    step_size: float
    target_accept: float = 0.8
    max_energy_error: float = 1000.0

def numerical_gradient(f: Callable[[Array], float], x: Array, eps: float = 1e-5) -> Array:
    """Simple central difference gradient."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = f(x_plus)
        
        x_minus = x.copy()
        x_minus[i] -= eps
        f_minus = f(x_minus)
        
        grad[i] = (f_plus - f_minus) / (2.0 * eps)
    return grad

def leapfrog(
    theta: Array,
    momentum: Array,
    step_size: float,
    n_steps: int,
    grad_potential_fn: Callable[[Array], Array],
    inverse_mass_matrix: Array,
) -> tuple[Array, Array]:
    """Leapfrog integrator for HMC."""
    theta_new = theta.copy()
    momentum_new = momentum.copy()

    # Half momentum update
    grad = grad_potential_fn(theta_new)
    momentum_new -= 0.5 * step_size * grad

    for i in range(n_steps):
        # Full position update
        theta_new += step_size * (inverse_mass_matrix * momentum_new)
        
        # Make sure theta_new is valid, if potential is inf, grad will be bad
        grad = grad_potential_fn(theta_new)
        
        if i != n_steps - 1:
            # Full momentum update
            momentum_new -= step_size * grad

    # Final half momentum update
    momentum_new -= 0.5 * step_size * grad
    
    # Optional: negate momentum for symmetry (not strictly required since kinetic energy is symmetric)
    # momentum_new = -momentum_new
    
    return theta_new, momentum_new

class HamiltonianMonteCarlo(MCMCKernel):
    def __init__(
        self,
        model: SpectralModel,
        beta: float,
        rng: np.random.Generator,
        config: HMCConfig,
        mass_matrix: Optional[Array] = None
    ):
        self.model = model
        self.beta = float(beta)
        self.rng = rng
        self.config = config
        
        dim = self.model.parameterization.ndim
        if mass_matrix is None:
            mass_matrix = np.ones(dim)
        self.mass_matrix = np.asarray(mass_matrix, dtype=float)
        self.inverse_mass_matrix = 1.0 / self.mass_matrix

    def potential_energy(self, theta: Array) -> float:
        # U_beta(theta) = -log q_beta(theta)
        # Note: log_tempered_target returns q_beta(theta) up to a constant.
        # So potential is just -log_tempered_target
        target = self.model.log_tempered_target(theta, self.beta)
        if not np.isfinite(target):
            return np.inf
        return -target

    def grad_potential_energy(self, theta: Array) -> Array:
        # Evaluate numerical gradient of potential
        return numerical_gradient(self.potential_energy, theta)
        
    def kinetic_energy(self, momentum: Array) -> float:
        return 0.5 * np.sum(momentum**2 * self.inverse_mass_matrix)

    def step(self, state: MetropolisState) -> MetropolisState:
        # 1. Sample fresh momentum
        momentum = self.rng.normal(0, np.sqrt(self.mass_matrix))
        
        current_U = self.potential_energy(state.theta)
        current_K = self.kinetic_energy(momentum)
        
        # 2. Integrate dynamics
        try:
            proposal_theta, proposal_momentum = leapfrog(
                state.theta,
                momentum,
                self.config.step_size,
                self.config.num_steps,
                self.grad_potential_energy,
                self.inverse_mass_matrix
            )
        except Exception:
            # Numerical failure in gradient evaluation or leaps
            proposal_theta = state.theta
            proposal_U = np.inf
            proposal_K = 0.0
        else:
            proposal_U = self.potential_energy(proposal_theta)
            proposal_K = self.kinetic_energy(proposal_momentum)

        # 3. Metropolis correction
        # H_new = U_new + K_new, H_old = U_old + K_old
        # log_alpha = -H_new + H_old
        
        if not np.isfinite(proposal_U) or not np.isfinite(proposal_K):
            delta_h = np.inf
        else:
            delta_h = (proposal_U + proposal_K) - (current_U + current_K)
            
        # Check for divergence
        if np.abs(delta_h) > self.config.max_energy_error:
            # Divergence, reject automatically
            delta_h = np.inf

        log_alpha = min(0.0, -delta_h)
        accept = np.log(self.rng.random()) < log_alpha

        state.attempted += 1
        if accept:
            state.theta = proposal_theta
            state.log_target = -proposal_U
            state.energy = self.model.energy(proposal_theta)
            state.accepted += 1
            
        return state

def hmc_kernel_factory(model: SpectralModel, beta: float, rng: np.random.Generator, config: Optional[HMCConfig] = None) -> HamiltonianMonteCarlo:
    if config is None:
        # Default educational configuration
        config = HMCConfig(num_steps=10, step_size=0.01)
    return HamiltonianMonteCarlo(model, beta, rng, config)
