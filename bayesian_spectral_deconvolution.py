"""
Bayesian spectral deconvolution with exchange Monte Carlo (parallel tempering).

Educational implementation based on:
    Nagata, K., Sugita, S., & Okada, M. (2012),
    "Bayesian spectral deconvolution with the exchange Monte Carlo method",
    Neural Networks 28, 82-89.

This file is intentionally verbose. It is designed for learning rather than
maximum performance. It implements:
  1. Gaussian RBF spectral model
  2. Bayesian posterior with the paper's priors
  3. Random-walk Metropolis within each temperature
  4. Exchange / parallel-tempering swaps between adjacent temperatures
  5. Marginal-likelihood estimation using the paper's product identity
  6. Model selection over K
  7. A small synthetic demonstration and sanity checks

No external spectral data are required for the main demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt


Array = np.ndarray


# -----------------------------------------------------------------------------
# Basic numerical helpers
# -----------------------------------------------------------------------------

def logsumexp(values: Array) -> float:
    """Stable log(sum(exp(values)))."""
    values = np.asarray(values, dtype=float)
    m = np.max(values)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(values - m))))


def logmeanexp(values: Array) -> float:
    """Stable log(mean(exp(values)))."""
    values = np.asarray(values, dtype=float)
    return logsumexp(values) - np.log(values.size)


# -----------------------------------------------------------------------------
# Model: sum of Gaussian radial basis functions
# -----------------------------------------------------------------------------

def gaussian_rbf(x: Array, mu: float, b: float) -> Array:
    """
    Gaussian basis used in Eq. (2):

        phi(x) = exp[- b/2 * (x - mu)^2]

    Notice that b is a *precision-like bandwidth parameter*: larger b means
    a narrower Gaussian, because the exponent becomes more negative faster.
    """
    return np.exp(-0.5 * b * (x - mu) ** 2)


def rbf_spectrum(x: Array, theta: Array) -> Array:
    """
    Evaluate the model

        f(x; theta) = sum_k a_k phi_k(x)

    theta is stored as [a_1,...,a_K, mu_1,...,mu_K, b_1,...,b_K].
    """
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    K = theta.size // 3
    a = theta[:K]
    mu = theta[K:2 * K]
    b = theta[2 * K:]

    # Shape: (K, n)
    basis = np.exp(-0.5 * b[:, None] * (x[None, :] - mu[:, None]) ** 2)
    return a @ basis


def mean_squared_error(x: Array, y: Array, theta: Array) -> float:
    """Paper's E(theta) in Eq. (3): 1/(2n) * sum squared residuals."""
    residual = y - rbf_spectrum(x, theta)
    return float(0.5 * np.mean(residual ** 2))


def gaussian_mgm(x: Array, theta: Array) -> Array:
    """Modified Gaussian model from Eq. (21), for log reflectance."""
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    K = (theta.size - 2) // 3
    c0, c1 = theta[-2:]
    a = theta[:K]
    mu = theta[K:2 * K]
    b = theta[2 * K:3 * K]
    bands = np.sum(
        a[:, None] * np.exp(-0.5 * b[:, None] * (x[None, :] - mu[:, None]) ** 2),
        axis=0,
    )
    return c0 + c1 / x + bands


# -----------------------------------------------------------------------------
# Priors from the paper
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticPrior:
    """Hyperparameters in Section 3.1 of the paper."""
    eta_a: float = 5.0
    lambda_a: float = 5.0
    nu0: float = 1.5
    xi0: float = 5.0
    eta_b: float = 5.0
    lambda_b: float = 0.04


@dataclass(frozen=True)
class MGMRawPrior:
    """Hyperparameters in Section 3.2 for the olivine MGM experiment."""
    eta_a: float = 3.0
    lambda_a: float = 2.0
    nu0: float = 1.25
    xi0: float = 2.5
    eta_b: float = 5.0
    lambda_b: float = 0.04
    eta_c: float = 1.0
    lambda_c: float = 10.0


def log_gamma_pdf_positive(x: float, shape: float, rate: float) -> float:
    """Log Gamma(shape, rate) density, x>0."""
    if x <= 0:
        return -np.inf
    return (
        shape * np.log(rate)
        - lgamma(shape)
        + (shape - 1.0) * np.log(x)
        - rate * x
    )


def log_normal_pdf(x: float, mean: float, precision: float) -> float:
    """Log N(mean, precision^{-1}) density."""
    return 0.5 * np.log(precision / (2.0 * np.pi)) - 0.5 * precision * (x - mean) ** 2


def log_prior_theta(theta: Array, K: int, prior: SyntheticPrior) -> float:
    """
    Sum log priors for {a_k, mu_k, b_k}.

    This is exactly the factorized prior form implied by Eqs. (17)-(19).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.size != 3 * K:
        raise ValueError("theta must contain 3*K entries")

    a = theta[:K]
    mu = theta[K:2 * K]
    b = theta[2 * K:]

    value = 0.0
    for ak, muk, bk in zip(a, mu, b):
        value += log_gamma_pdf_positive(ak, prior.eta_a, prior.lambda_a)
        value += log_normal_pdf(muk, prior.nu0, prior.xi0)
        value += log_gamma_pdf_positive(bk, prior.eta_b, prior.lambda_b)
    return float(value)


# -----------------------------------------------------------------------------
# Bayesian target density q(theta; beta)
# -----------------------------------------------------------------------------

@dataclass
class BayesianRBFProblem:
    x: Array
    y: Array
    sigma2: float
    K: int
    prior: SyntheticPrior = SyntheticPrior()

    def energy(self, theta: Array) -> float:
        return mean_squared_error(self.x, self.y, theta)

    def log_likelihood(self, theta: Array) -> float:
        """
        Log likelihood up to the Gaussian normalizing constant.

        For a fixed sigma^2, the paper's q(theta; beta) only needs
            -(n/sigma^2) beta E(theta) + log prior.
        """
        n = self.x.size
        return -(n / self.sigma2) * self.energy(theta)

    def log_target(self, theta: Array, beta: float) -> float:
        """Log of q(theta; beta), up to a beta-dependent normalization constant."""
        lp = log_prior_theta(theta, self.K, self.prior)
        if not np.isfinite(lp):
            return -np.inf
        return beta * self.log_likelihood(theta) + lp


# -----------------------------------------------------------------------------
# Temperature ladder from the paper
# -----------------------------------------------------------------------------

def paper_beta_schedule(L: int) -> Array:
    """
    Temperature / inverse-temperature ladder used in Section 3.1:

        beta_1 = 0
        beta_l = 1.5 ** (l-L), l>1

    where L=24 in the synthetic experiment.
    """
    if L < 2:
        raise ValueError("Need at least two temperatures")
    beta = np.empty(L, dtype=float)
    beta[0] = 0.0
    for l in range(1, L):
        # l in the paper is 1-indexed. Here index l is 0-indexed, so paper l=l+1.
        paper_l = l + 1
        beta[l] = 1.5 ** (paper_l - L)
    beta[-1] = 1.0
    return beta


# -----------------------------------------------------------------------------
# Random-walk Metropolis sampler for one temperature
# -----------------------------------------------------------------------------

@dataclass
class MetropolisState:
    theta: Array
    log_target: float
    accepted: int = 0
    attempted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempted if self.attempted else np.nan


class RandomWalkMetropolis:
    """
    Simple Gaussian random-walk Metropolis sampler.

    Proposal:
        theta' = theta + Normal(0, proposal_std^2)

    The proposal is symmetric, so the proposal densities cancel and
    acceptance is min(1, exp(log_target' - log_target)).
    """

    def __init__(
        self,
        problem: BayesianRBFProblem,
        beta: float,
        rng: np.random.Generator,
        proposal_scales: Optional[Array] = None,
    ) -> None:
        self.problem = problem
        self.beta = float(beta)
        self.rng = rng
        if proposal_scales is None:
            # Crude defaults that work for the educational demo. In a serious
            # implementation, tune these using pilot acceptance rates.
            K = problem.K
            proposal_scales = np.concatenate([
                np.full(K, 0.03),   # strengths
                np.full(K, 0.02),   # centers
                np.full(K, 2.0),    # b parameters
            ])
        self.proposal_scales = np.asarray(proposal_scales, dtype=float)

    def step(self, state: MetropolisState) -> MetropolisState:
        proposal = state.theta + self.rng.normal(0.0, self.proposal_scales)
        proposal_log_target = self.problem.log_target(proposal, self.beta)

        log_alpha = proposal_log_target - state.log_target
        accept = np.log(self.rng.random()) < min(0.0, log_alpha)

        state.attempted += 1
        if accept:
            state.theta = proposal
            state.log_target = proposal_log_target
            state.accepted += 1
        return state


# -----------------------------------------------------------------------------
# Exchange Monte Carlo / parallel tempering
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Marginal likelihood / evidence from the paper's product identity
# -----------------------------------------------------------------------------

@dataclass
class EvidenceEstimate:
    log_z: float
    log_ratios: Array
    ratio_standard_errors: Array


def estimate_log_evidence_from_exchange(
    problem: BayesianRBFProblem,
    result: ExchangeResult,
) -> EvidenceEstimate:
    """
    Estimate log Z(1) using Eq. (10).

    For each adjacent pair:

        Z(beta_{l+1}) / Z(beta_l)
          = E_{q_beta_l}[ exp(-n/sigma^2 * (beta_{l+1}-beta_l) E(theta)) ]

    We use the samples produced at beta_l and compute the expectation by
    ordinary Monte Carlo averaging.
    """
    n = problem.x.size
    sigma2 = problem.sigma2
    log_ratios = []
    ratio_se = []

    for l in range(result.beta.size - 1):
        delta_beta = result.beta[l + 1] - result.beta[l]
        energies = result.energy_trace_by_temperature[l]
        if energies.size == 0:
            raise ValueError("No samples available for evidence estimation")

        log_weights = -(n / sigma2) * delta_beta * energies
        log_ratio = logmeanexp(log_weights)
        log_ratios.append(log_ratio)

        # Monte Carlo standard error on the ratio itself.
        weights = np.exp(log_weights - np.max(log_weights))
        weights *= np.exp(np.max(log_weights))
        ratio = np.mean(weights)
        if energies.size > 1:
            se = np.std(weights, ddof=1) / np.sqrt(energies.size)
        else:
            se = np.nan
        ratio_se.append(se / max(ratio, 1e-300))

    log_ratios = np.asarray(log_ratios)
    return EvidenceEstimate(
        log_z=float(np.sum(log_ratios)),
        log_ratios=log_ratios,
        ratio_standard_errors=np.asarray(ratio_se),
    )


# -----------------------------------------------------------------------------
# Model selection over K
# -----------------------------------------------------------------------------

@dataclass
class ModelRun:
    K: int
    problem: BayesianRBFProblem
    exchange_result: ExchangeResult
    evidence: EvidenceEstimate

    @property
    def stochastic_complexity(self) -> float:
        return -self.evidence.log_z

    @property
    def posterior_mode_sample(self) -> Array:
        """
        Approximate MAP using the beta=1 samples.

        The paper reports estimating theta-hat that maximizes posterior
        probability after selecting K. This implementation approximates that
        by taking the sampled beta=1 state with largest log posterior.
        """
        samples = self.exchange_result.samples_by_temperature[-1]
        scores = np.asarray([
            self.problem.log_target(theta, beta=1.0) for theta in samples
        ])
        return samples[np.argmax(scores)]


def run_model_selection(
    x: Array,
    y: Array,
    sigma2: float,
    K_values: list[int],
    prior: SyntheticPrior,
    L: int = 24,
    burn_in: int = 2_000,
    expectation_steps: int = 2_000,
    seed: int = 1234,
) -> list[ModelRun]:
    """Run exchange MC independently for each candidate K."""
    runs = []
    master_rng = np.random.default_rng(seed)
    beta = paper_beta_schedule(L)

    for K in K_values:
        # Use a fresh deterministic stream per K for reproducibility.
        child_seed = int(master_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(child_seed)
        problem = BayesianRBFProblem(x=x, y=y, sigma2=sigma2, K=K, prior=prior)
        emc = ExchangeMonteCarlo(problem, beta=beta, rng=rng)
        result = emc.run(
            burn_in=burn_in,
            expectation_steps=expectation_steps,
            swap_every=1,
            record_every=1,
            store_state_trace=False,
        )
        evidence = estimate_log_evidence_from_exchange(problem, result)
        runs.append(ModelRun(K, problem, result, evidence))
        print(
            f"K={K:2d} | stochastic complexity -log Z = {runs[-1].stochastic_complexity: .3f} "
            f"| beta=1 MH acc={result.within_acceptance[-1]:.3f} "
            f"| mean swap acc={np.mean(result.exchange_acceptance):.3f}"
        )

    return runs


# -----------------------------------------------------------------------------
# Synthetic data from the paper (Section 3.1)
# -----------------------------------------------------------------------------

PAPER_AMPLITUDES = np.array([0.587, 1.522, 1.183])
PAPER_CENTERS = np.array([1.210, 1.455, 1.703])
PAPER_B = np.array([95.689, 146.837, 164.469])


def make_paper_like_synthetic_data(
    n_points: int = 301,
    sigma2: float = 0.01,
    seed: int = 7,
) -> tuple[Array, Array, Array]:
    """
    Generate the synthetic setting described in the paper.

    x = 0, 0.01, ..., 3.0  (301 points)
    sigma^2 = 0.01
    true model = 3 Gaussian bands with the paper's parameters.

    Returns x, noisy y, noiseless true y.
    """
    if n_points != 301:
        x = np.linspace(0.0, 3.0, n_points)
    else:
        x = np.arange(0.0, 3.0 + 1e-12, 0.01)
    theta_star = np.concatenate([PAPER_AMPLITUDES, PAPER_CENTERS, PAPER_B])
    y_true = rbf_spectrum(x, theta_star)
    rng = np.random.default_rng(seed)
    y = y_true + rng.normal(0.0, np.sqrt(sigma2), size=x.size)
    return x, y, y_true


# -----------------------------------------------------------------------------
# Helpful educational toy example: a bimodal posterior
# -----------------------------------------------------------------------------

def double_well_log_density(x: float) -> float:
    """A toy target with two separated modes, useful for seeing tempering."""
    # Unnormalized log density for a symmetric two-mode mixture.
    log1 = -0.5 * ((x + 4.0) / 0.65) ** 2
    log2 = -0.5 * ((x - 4.0) / 0.65) ** 2
    return logsumexp(np.array([log1, log2]))


def standard_mh_double_well(
    steps: int = 20_000,
    proposal_sd: float = 0.7,
    seed: int = 1,
    start: float = -4.0,
) -> Array:
    rng = np.random.default_rng(seed)
    x = float(start)
    trace = np.empty(steps)
    current = double_well_log_density(x)
    for t in range(steps):
        proposal = x + rng.normal(0, proposal_sd)
        lp = double_well_log_density(proposal)
        if np.log(rng.random()) < min(0.0, lp - current):
            x = proposal
            current = lp
        trace[t] = x
    return trace


def parallel_tempering_double_well(
    steps: int = 20_000,
    betas: Optional[Array] = None,
    proposal_sd: float = 0.9,
    seed: int = 2,
) -> Array:
    """
    Minimal 1-D exchange Monte Carlo example.

    This is not the spectral model; it isolates the *idea* of escaping local
    modes using temperatures.
    """
    rng = np.random.default_rng(seed)
    if betas is None:
        betas = np.array([0.02, 0.05, 0.12, 0.25, 0.5, 1.0])
    L = len(betas)
    states = rng.normal(0, 4, size=L)
    traces = np.empty((steps, L))

    def energy(z: float) -> float:
        # High density = low energy.
        return -double_well_log_density(z)

    for t in range(steps):
        # Within-temperature MH moves.
        for l, beta in enumerate(betas):
            old = states[l]
            new = old + rng.normal(0, proposal_sd)
            log_alpha = -beta * energy(new) + beta * energy(old)
            if np.log(rng.random()) < min(0.0, log_alpha):
                states[l] = new

        # Adjacent swaps.
        for l in range(0, L - 1):
            b1, b2 = betas[l], betas[l + 1]
            e1, e2 = energy(states[l]), energy(states[l + 1])
            log_alpha = (b2 - b1) * (e2 - e1)
            if np.log(rng.random()) < min(0.0, log_alpha):
                states[l], states[l + 1] = states[l + 1], states[l]

        traces[t] = states
    return traces


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_synthetic_data(x: Array, y: Array, y_true: Array, out: Path) -> None:
    theta_star = np.concatenate([PAPER_AMPLITUDES, PAPER_CENTERS, PAPER_B])
    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=8, alpha=0.45, label="observed data")
    plt.plot(x, y_true, linewidth=2, label="true sum of 3 Gaussians")
    for k in range(3):
        band = PAPER_AMPLITUDES[k] * gaussian_rbf(x, PAPER_CENTERS[k], PAPER_B[k])
        plt.plot(x, band, linestyle="--", label=f"true band {k+1}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Paper-like synthetic data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.show()


def plot_temperature_ladder(beta: Array, out: Path) -> None:
    T = np.where(beta > 0, 1.0 / beta, np.inf)
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(beta) + 1), beta, marker="o")
    plt.xlabel("replica index")
    plt.ylabel("inverse temperature beta")
    plt.title("Exchange Monte Carlo temperature ladder")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.show()


def plot_toy_mixing(out: Path) -> None:
    mh = standard_mh_double_well()
    pt = parallel_tempering_double_well()

    plt.figure(figsize=(9, 6))
    plt.plot(mh[:5000], linewidth=0.8, label="ordinary Metropolis")
    plt.plot(pt[:5000, -1], linewidth=0.8, label="exchange MC at beta=1")
    plt.axhline(0, linestyle=":")
    plt.xlabel("iteration")
    plt.ylabel("state")
    plt.title("Why tempering helps: escaping between two separated modes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.show()


def plot_model_selection(runs: list[ModelRun], out: Path) -> None:
    K = [r.K for r in runs]
    sc = [r.stochastic_complexity for r in runs]
    plt.figure(figsize=(8, 4.5))
    plt.plot(K, sc, marker="o")
    plt.xlabel("number of Gaussian bands K")
    plt.ylabel("stochastic complexity = -log Z")
    plt.title("Bayesian model selection via marginal likelihood")
    plt.xticks(K)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.show()


def plot_fit(run: ModelRun, x: Array, y: Array, out: Path) -> None:
    theta = run.posterior_mode_sample
    fit = rbf_spectrum(x, theta)
    K = run.K

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=8, alpha=0.35, label="observed")
    plt.plot(x, fit, linewidth=2, label=f"approximate MAP fit, K={K}")

    a = theta[:K]
    mu = theta[K:2 * K]
    b = theta[2 * K:]
    for k in range(K):
        plt.plot(
            x,
            a[k] * gaussian_rbf(x, mu[k], b[k]),
            linestyle="--",
            linewidth=1,
            label=f"band {k+1}",
        )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Selected model fit (K={K})")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.show()


def plot_exchange_diagnostics(run: ModelRun, out: Path) -> None:
    result = run.exchange_result
    # Plot only the hottest, a middle, and beta=1 replica.
    idx = [0, len(result.beta) // 2, len(result.beta) - 1]
    plt.figure(figsize=(9, 5))
    for i in idx:
        plt.plot(result.energy_trace_by_temperature[i], alpha=0.9,
                 label=f"beta={result.beta[i]:.3g}")
    plt.xlabel("recorded iteration")
    plt.ylabel("E(theta)")
    plt.title(f"Energy traces across temperatures, K={run.K}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.show()


def plot_parameter_histograms(run: ModelRun, out: Path) -> None:
    """Overlay parameter histograms from beta=1, keeping the script simple."""
    samples = run.exchange_result.samples_by_temperature[-1]
    K = run.K
    theta = run.posterior_mode_sample

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False)
    names = ["strength a", "center mu", "bandwidth b"]
    for row, start in enumerate([0, K, 2 * K]):
        for k in range(K):
            axes[row].hist(samples[:, start + k], bins=40, alpha=0.35,
                           label=f"k={k+1}")
            axes[row].axvline(theta[start + k], linestyle="--")
        axes[row].set_ylabel(names[row])
        axes[row].legend(ncol=min(K, 4))
    axes[-1].set_xlabel("parameter value")
    fig.suptitle(f"Approximate posterior samples at beta=1, K={K}")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.show()


# -----------------------------------------------------------------------------
# Sanity tests
# -----------------------------------------------------------------------------

def run_sanity_tests() -> None:
    """Small tests that catch common implementation mistakes."""
    x = np.linspace(0, 3, 31)
    theta = np.array([0.7, 1.2, 1.0, 1.8, 40.0, 80.0])
    y = rbf_spectrum(x, theta)
    prior = SyntheticPrior()
    problem = BayesianRBFProblem(x, y, sigma2=0.01, K=2, prior=prior)

    # Test 1: vectorized RBF agrees with a direct loop.
    direct = np.zeros_like(x)
    for a, mu, b in zip(theta[:2], theta[2:4], theta[4:]):
        direct += a * np.exp(-0.5 * b * (x - mu) ** 2)
    np.testing.assert_allclose(y, direct, rtol=1e-12, atol=1e-12)

    # Test 2: paper beta schedule starts at 0 and ends at 1.
    beta = paper_beta_schedule(24)
    assert beta[0] == 0.0
    assert np.isclose(beta[-1], 1.0)
    assert np.all(np.diff(beta[1:]) > 0)

    # Test 3: the swap exponent is zero when energies are equal.
    E = problem.energy(theta)
    beta1, beta2 = 0.2, 0.8
    log_v = (x.size / problem.sigma2) * (beta2 - beta1) * (E - E)
    assert np.isclose(log_v, 0.0)

    # Test 4: lower energy at the colder temperature should be favored.
    E_cold, E_hot = 0.1, 1.0
    log_v = (x.size / problem.sigma2) * (beta2 - beta1) * (E_hot - E_cold)
    assert log_v > 0.0

    # Test 5: q(beta=0) is exactly prior-only apart from normalization.
    theta2 = theta.copy()
    theta2[0] += 0.1
    assert np.isclose(
        problem.log_target(theta, 0.0) - problem.log_target(theta2, 0.0),
        log_prior_theta(theta, 2, prior) - log_prior_theta(theta2, 2, prior),
    )

    print("All sanity tests passed.")


# -----------------------------------------------------------------------------
# Main educational demo
# -----------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("bayesian_spectral_demo")
    out_dir.mkdir(exist_ok=True)

    print("Running sanity tests...")
    run_sanity_tests()

    print("\nGenerating paper-like synthetic data...")
    x, y, y_true = make_paper_like_synthetic_data(seed=7)
    plot_synthetic_data(x, y, y_true, out_dir / "01_synthetic_data.png")

    beta = paper_beta_schedule(24)
    plot_temperature_ladder(beta, out_dir / "02_temperature_ladder.png")
    plot_toy_mixing(out_dir / "03_toy_tempering_mixing.png")

    print("\nRunning model selection for K=1,...,4.")
    print("The original paper used K=1,...,8 and much longer chains; this demo is intentionally smaller.")
    prior = SyntheticPrior()
    runs = run_model_selection(
        x=x,
        y=y,
        sigma2=0.01,
        K_values=[1, 2, 3, 4],
        prior=prior,
        # Smaller than the paper for a quick educational run.
        L=12,
        burn_in=600,
        expectation_steps=600,
        seed=123,
    )

    plot_model_selection(runs, out_dir / "04_model_selection.png")

    best = min(runs, key=lambda r: r.stochastic_complexity)
    print(f"\nSelected K = {best.K}")
    theta_hat = best.posterior_mode_sample
    print("Approximate MAP/sample parameters:")
    print("  a  =", theta_hat[:best.K])
    print("  mu =", theta_hat[best.K:2 * best.K])
    print("  b  =", theta_hat[2 * best.K:])

    plot_fit(best, x, y, out_dir / "05_selected_fit.png")
    plot_exchange_diagnostics(best, out_dir / "06_exchange_diagnostics.png")
    plot_parameter_histograms(best, out_dir / "07_parameter_histograms.png")

    print("\nWrote plots to:", out_dir.resolve())
    print("Model-selection summary:")
    for r in runs:
        print(f"  K={r.K}: -log Z = {r.stochastic_complexity:.3f}")


if __name__ == "__main__":
    main()
