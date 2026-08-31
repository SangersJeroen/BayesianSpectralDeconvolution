import numpy as np
from typing import Optional
from dataclasses import dataclass

from .bayesian_rbf import BayesianRBFProblem
from .samplers import RandomWalkMetropolis, MetropolisState

Array = np.ndarray

@dataclass
class ExchangeResult:
    beta: Array
    samples_by_temperature: list[Array]
    energy_trace_by_temperature: list[Array]
    within_acceptance: Array
    exchange_acceptance: Array
    all_states_trace: Optional[Array] = None


class ExchangeMonteCarlo:
    """
    Educational implementation of the paper's exchange Monte Carlo method.

    At each iteration:
      1. Perform one ordinary Metropolis update in every replica.
      2. Attempt swaps between adjacent beta values.

    The swap acceptance probability is Eq. (13)-(15):

        alpha = min(1,
            exp[(n/sigma^2)(beta_{l+1}-beta_l)
                (E(theta_{l+1}) - E(theta_l))]
        )

    The interesting idea is that a state trapped near a local minimum at beta=1
    can swap with a high-temperature state. That lets it explore a much wider
    region of parameter space before returning to beta=1.
    """

    def __init__(
        self,
        problem: BayesianRBFProblem,
        beta: Array,
        rng: np.random.Generator,
        proposal_scales: Optional[Array] = None,
    ) -> None:
        self.problem = problem
        self.beta = np.asarray(beta, dtype=float)
        self.rng = rng
        self.L = self.beta.size
        self.samplers = [
            RandomWalkMetropolis(problem, b, rng, proposal_scales)
            for b in self.beta
        ]

    def initial_states(self) -> list[MetropolisState]:
        """Initialize every replica from the prior, as done in the paper."""
        K = self.problem.K
        prior = self.problem.prior
        states: list[MetropolisState] = []
        for _ in range(self.L):
            a = self.rng.gamma(shape=prior.eta_a, scale=1.0 / prior.lambda_a, size=K)
            mu = self.rng.normal(loc=prior.nu0, scale=1.0 / np.sqrt(prior.xi0), size=K)
            b = self.rng.gamma(shape=prior.eta_b, scale=1.0 / prior.lambda_b, size=K)
            theta = np.concatenate([a, mu, b])
            log_target = self.problem.log_target(theta, 0.0)
            states.append(MetropolisState(theta, log_target))
        return states

    def _attempt_swap(self, states: list[MetropolisState], l: int) -> bool:
        """Attempt exchange between replica l and l+1, using Eq. (15)."""
        s1 = states[l]
        s2 = states[l + 1]
        beta1 = self.beta[l]
        beta2 = self.beta[l + 1]

        E1 = self.problem.energy(s1.theta)
        E2 = self.problem.energy(s2.theta)
        n = self.problem.x.size
        sigma2 = self.problem.sigma2

        log_v = (n / sigma2) * (beta2 - beta1) * (E2 - E1)
        accept = np.log(self.rng.random()) < min(0.0, log_v)

        if accept:
            # IMPORTANT: swap the states, not the temperatures.
            # Recompute their log targets because each theta is now associated
            # with a different beta.
            theta1 = s1.theta.copy()
            theta2 = s2.theta.copy()
            s1.theta = theta2
            s2.theta = theta1
            s1.log_target = self.problem.log_target(s1.theta, beta1)
            s2.log_target = self.problem.log_target(s2.theta, beta2)
        return bool(accept)

    def run(
        self,
        burn_in: int,
        expectation_steps: int,
        swap_every: int = 1,
        record_every: int = 1,
        store_state_trace: bool = False,
    ) -> ExchangeResult:
        """Run burn-in, then collect posterior samples at every temperature."""
        if burn_in < 0 or expectation_steps <= 0:
            raise ValueError("burn_in >= 0 and expectation_steps > 0 are required")

        states = self.initial_states()

        within_attempts = np.zeros(self.L, dtype=int)
        within_accepts = np.zeros(self.L, dtype=int)
        exchange_attempts = np.zeros(self.L - 1, dtype=int)
        exchange_accepts = np.zeros(self.L - 1, dtype=int)

        def one_step(step_number: int) -> None:
            # 1. Within-temperature MCMC updates.
            for l, sampler in enumerate(self.samplers):
                before_attempts = states[l].attempted
                before_accepts = states[l].accepted
                sampler.step(states[l])
                within_attempts[l] += states[l].attempted - before_attempts
                within_accepts[l] += states[l].accepted - before_accepts

            # 2. Adjacent exchange attempts.
            if step_number % swap_every == 0:
                # Alternating parity avoids always attempting the same pair first.
                parity = (step_number // swap_every) % 2
                for l in range(parity, self.L - 1, 2):
                    exchange_attempts[l] += 1
                    if self._attempt_swap(states, l):
                        exchange_accepts[l] += 1

        # Burn-in
        for step in range(1, burn_in + 1):
            one_step(step)

        samples_by_temperature = [[] for _ in range(self.L)]
        energy_trace_by_temperature = [[] for _ in range(self.L)]
        raw_state_trace = []

        # Expectation phase
        for step in range(1, expectation_steps + 1):
            one_step(burn_in + step)
            if step % record_every == 0:
                if store_state_trace:
                    raw_state_trace.append(np.stack([s.theta.copy() for s in states]))
                for l, state in enumerate(states):
                    samples_by_temperature[l].append(state.theta.copy())
                    energy_trace_by_temperature[l].append(self.problem.energy(state.theta))

        return ExchangeResult(
            beta=self.beta.copy(),
            samples_by_temperature=[np.asarray(s) for s in samples_by_temperature],
            energy_trace_by_temperature=[np.asarray(s) for s in energy_trace_by_temperature],
            within_acceptance=within_accepts / np.maximum(within_attempts, 1),
            exchange_acceptance=exchange_accepts / np.maximum(exchange_attempts, 1),
            all_states_trace=np.asarray(raw_state_trace) if store_state_trace else None,
        )
