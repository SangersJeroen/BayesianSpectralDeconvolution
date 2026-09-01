import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
from .models import SpectralModel
from .samplers.metropolis import MetropolisState, MCMCKernel

Array = np.ndarray

@dataclass
class ExchangeResult:
    beta: Array
    samples_by_temperature: list[Array]
    energy_trace_by_temperature: list[Array]
    within_acceptance: Array
    exchange_acceptance: Array
    all_states_trace: Optional[Array] = None

class ParallelTempering:
    def __init__(
        self,
        model: SpectralModel,
        betas: Array,
        rng: np.random.Generator,
        kernel_factory: Callable[[SpectralModel, float, np.random.Generator], MCMCKernel]
    ):
        self.model = model
        self.beta = np.asarray(betas, dtype=float)
        self.rng = rng
        self.L = self.beta.size
        self.samplers = [
            kernel_factory(model, b, rng)
            for b in self.beta
        ]

    def initial_states(self) -> list[MetropolisState]:
        states = []
        for b in self.beta:
            theta = self.model.prior.sample(self.rng, self.model.parameterization)
            log_target = self.model.log_tempered_target(theta, b)
            energy = self.model.energy(theta)
            states.append(MetropolisState(theta, log_target, energy))
        return states

    def _attempt_swap(self, states: list[MetropolisState], l: int) -> bool:
        s1 = states[l]
        s2 = states[l + 1]
        beta1 = self.beta[l]
        beta2 = self.beta[l + 1]

        E1 = s1.energy
        E2 = s2.energy
        n = self.model.n
        
        # We need sigma2 for paper's exact swap logic:
        # log_v = (n / sigma2) * (beta2 - beta1) * (E2 - E1)
        if hasattr(self.model.likelihood, 'sigma2'):
            sigma2 = self.model.likelihood.sigma2
            log_v = (n / sigma2) * (beta2 - beta1) * (E2 - E1)
        else:
            # Fallback if no sigma2: the exchange log_v is generally:
            # log_target(theta2, beta1) + log_target(theta1, beta2) - log_target(theta1, beta1) - log_target(theta2, beta2)
            # which simplifies to (beta2 - beta1) * (log_likelihood(theta1) - log_likelihood(theta2))
            ll1 = self.model.log_likelihood(s1.theta)
            ll2 = self.model.log_likelihood(s2.theta)
            log_v = (beta2 - beta1) * (ll1 - ll2)

        accept = np.log(self.rng.random()) < min(0.0, float(log_v))

        if accept:
            # Swap states
            theta1, e1 = s1.theta.copy(), s1.energy
            theta2, e2 = s2.theta.copy(), s2.energy
            
            s1.theta, s1.energy = theta2, e2
            s2.theta, s2.energy = theta1, e1
            
            # Recompute log targets for new betas
            s1.log_target = self.model.log_tempered_target(s1.theta, beta1)
            s2.log_target = self.model.log_tempered_target(s2.theta, beta2)
            
        return bool(accept)

    def run(
        self,
        burn_in: int,
        samples: int,
        swap_every: int = 1,
        record_every: int = 1,
        store_state_trace: bool = False,
    ) -> ExchangeResult:
        states = self.initial_states()

        within_attempts = np.zeros(self.L, dtype=int)
        within_accepts = np.zeros(self.L, dtype=int)
        exchange_attempts = np.zeros(self.L - 1, dtype=int)
        exchange_accepts = np.zeros(self.L - 1, dtype=int)

        def one_step(step_number: int):
            for l, sampler in enumerate(self.samplers):
                before_attempts = states[l].attempted
                before_accepts = states[l].accepted
                self.samplers[l].step(states[l])
                within_attempts[l] += states[l].attempted - before_attempts
                within_accepts[l] += states[l].accepted - before_accepts

            if step_number % swap_every == 0:
                parity = (step_number // swap_every) % 2
                for l in range(parity, self.L - 1, 2):
                    exchange_attempts[l] += 1
                    if self._attempt_swap(states, l):
                        exchange_accepts[l] += 1

        for step in range(1, burn_in + 1):
            one_step(step)

        samples_by_temperature = [[] for _ in range(self.L)]
        energy_trace_by_temperature = [[] for _ in range(self.L)]
        raw_state_trace = []

        for step in range(1, samples + 1):
            one_step(burn_in + step)
            if step % record_every == 0:
                if store_state_trace:
                    raw_state_trace.append(np.stack([s.theta.copy() for s in states]))
                for l, state in enumerate(states):
                    samples_by_temperature[l].append(state.theta.copy())
                    energy_trace_by_temperature[l].append(state.energy)

        return ExchangeResult(
            beta=self.beta.copy(),
            samples_by_temperature=[np.asarray(s) for s in samples_by_temperature],
            energy_trace_by_temperature=[np.asarray(s) for s in energy_trace_by_temperature],
            within_acceptance=within_accepts / np.maximum(within_attempts, 1),
            exchange_acceptance=exchange_accepts / np.maximum(exchange_attempts, 1),
            all_states_trace=np.asarray(raw_state_trace) if store_state_trace else None,
        )
