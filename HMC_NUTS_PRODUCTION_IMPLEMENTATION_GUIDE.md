# Production Implementation Guide: HMC and NUTS for Bayesian Spectral Deconvolution

## 0. Purpose and scope

This document is a coding-agent handoff for upgrading the Bayesian spectral deconvolution codebase from random-walk Metropolis to production-quality Hamiltonian Monte Carlo (HMC) and No-U-Turn Sampling (NUTS), while preserving the paper's parallel-tempering / exchange-Monte-Carlo architecture.

The intended end state is:

```text
Basis + Prior + Likelihood
          |
          v
  differentiable log-density
          |
          v
   HMC / NUTS kernels
          |
          v
 per-temperature kernels
          |
          v
 parallel tempering / exchange
          |
          v
 posterior samples + evidence
          |
          v
 model selection over K
```

The implementation should be modular enough that:

- a new basis function does not require sampler changes;
- a new prior does not require sampler changes;
- HMC can be tested independently of NUTS;
- NUTS can be tested independently of parallel tempering;
- parallel tempering can use Metropolis, HMC, or NUTS as its within-temperature kernel;
- the model can expose gradients through automatic differentiation;
- all numerical and statistical diagnostics are explicit.

The motivating paper uses Gaussian/RBF basis functions, Gaussian observation noise, Bayesian evidence via a temperature path, and exchange Monte Carlo. Its inner sampler is conventional MCMC rather than HMC/NUTS. This guide specifies a modern sampler upgrade, not a claim about what the paper itself implemented.

---

# 1. Core design principles

## 1.1 Separate the model from the sampler

The sampler must never know that the scientific model is spectral deconvolution.

The sampler should only require an interface approximately equivalent to:

```python
log_density(theta, context) -> float
grad_log_density(theta, context) -> Array
```

or, preferably for JAX:

```python
log_density(theta, context) -> scalar
grad_log_density = jax.grad(log_density)
```

The spectral model should own:

- forward prediction;
- likelihood;
- priors;
- parameter transformations;
- temperature-scaled target density.

The sampler should own:

- momentum;
- leapfrog integration;
- Hamiltonian evaluation;
- Metropolis correction;
- NUTS tree building;
- adaptation;
- diagnostics.

Parallel tempering should own:

- temperature ladder;
- replica state;
- adjacent exchange;
- replica bookkeeping.

Evidence estimation should consume energy/sample outputs and should not know how samples were generated.

---

# 2. Recommended package structure

Refactor toward:

```text
bayesian_spectral/
├── pyproject.toml
├── README.md
├── src/
│   └── bayesian_spectral/
│       ├── __init__.py
│       ├── types.py
│       ├── basis.py
│       ├── priors.py
│       ├── transforms.py
│       ├── likelihoods.py
│       ├── models.py
│       ├── logdensity.py
│       ├── hmc.py
│       ├── nuts.py
│       ├── adaptation.py
│       ├── tempering.py
│       ├── evidence.py
│       ├── diagnostics.py
│       ├── initialization.py
│       └── utils.py
├── tests/
│   ├── test_basis.py
│   ├── test_priors.py
│   ├── test_transforms.py
│   ├── test_logdensity.py
│   ├── test_hmc.py
│   ├── test_nuts.py
│   ├── test_tempering.py
│   ├── test_evidence.py
│   └── test_end_to_end.py
├── benchmarks/
│   ├── benchmark_logdensity.py
│   ├── benchmark_hmc.py
│   └── benchmark_tempering.py
└── notebooks/
    ├── 01_metropolis_reference.ipynb
    ├── 02_hmc.ipynb
    └── 03_nuts_tempering.ipynb
```

Prefer small immutable result/config dataclasses over loosely structured dictionaries.

---

# 3. Mathematical target distribution

For one candidate model with parameter vector theta, the paper uses:

\[
p(\theta \mid D) \propto
\exp\left[-\frac{n}{\sigma^2}E(\theta)\right]\phi(\theta)
\]

where:

- \(E(\theta)\) is the model error;
- \(\phi(\theta)\) is the prior;
- \(n\) is the number of observations;
- \(\sigma^2\) is the observation-noise variance.

For exchange Monte Carlo, define:

\[
q_\beta(\theta)
\propto
\exp\left[-\frac{n}{\sigma^2}\beta E(\theta)\right]\phi(\theta)
\]

for \(0 \leq \beta \leq 1\).

The target at beta=1 is the posterior. The target at beta=0 is the prior.

The HMC/NUTS potential energy should be:

\[
U_\beta(\theta)=-\log q_\beta(\theta)
\]

up to an additive constant independent of theta:

\[
\boxed{
U_\beta(\theta)=
\frac{n}{\sigma^2}\beta E(\theta)-\log\phi(\theta)
}
\]

and:

\[
\boxed{
\nabla U_\beta(\theta)=
\frac{n}{\sigma^2}\beta\nabla E(\theta)-\nabla\log\phi(\theta)
}
\]

The sampler should work with `potential_energy(theta, beta)` and its gradient, preferably in unconstrained coordinates.

---

# 4. Parameterization and constrained parameters

HMC requires a smooth parameter space. Do not use hard rejection at constraints such as:

```python
if width <= 0:
    return -np.inf
```

for the actual HMC state whenever avoidable.

For strictly positive parameters such as Gaussian width/precision or positive amplitudes, use an unconstrained variable.

For example:

\[
z\in\mathbb R,
\qquad s=\exp(z)
\]

or:

\[
s=\operatorname{softplus}(z).
\]

If the transformation is part of the sampled-density change of variables, include the log absolute Jacobian determinant.

For:

\[
s=\exp(z)
\]

the Jacobian is:

\[
\left|\frac{ds}{dz}\right|=e^z=s
\]

and:

\[
\log |J| = z.
\]

The transformed target becomes:

\[
\log p(z \mid D)=
\log p(s(z)\mid D)+\log|J(z)|.
\]

Do not add a Jacobian if the prior has intentionally been defined directly on the unconstrained variable. Be explicit about which space the prior is specified in.

## Ordered peak centers

Label switching produces equivalent modes because:

\[
G_1+G_2+\cdots+G_K
\]

is invariant to permutation.

Do not impose:

\[
\mu_1<\mu_2<\cdots<\mu_K
\]

silently. Ordering is a model/parameterization decision and changes the integration domain. For exact reproduction of a symmetric prior/evidence formulation, keep the symmetric parameterization and handle label switching during post-processing.

If an ordered parameterization is later introduced, derive and test its Jacobian carefully.

---

# 5. Log-density interface

Create a sampler-facing interface such as:

```python
class DifferentiableTarget(Protocol):
    def log_density(self, theta: Array, beta: float) -> Scalar:
        ...

    def grad_log_density(self, theta: Array, beta: float) -> Array:
        ...
```

Prefer representing the target in terms of log density, not raw probability.

Required properties:

1. `log_density` must return finite values for valid states.
2. Invalid states should be represented in a controlled way.
3. Gradients should be finite for valid states.
4. The same function must be usable during warmup and production.
5. No hidden mutable state should affect density evaluation.
6. The function must be deterministic given `(theta, beta, data)`.

For JAX, make model functions pure and JIT-compatible.

---

# 6. Hamiltonian Monte Carlo

## 6.1 Hamiltonian

Introduce momentum \(p\).

For a Euclidean metric with mass matrix \(M\):

\[
K(p)=
\frac12 p^T M^{-1}p
+
\frac12\log |2\pi M|.
\]

The additive normalizing constant can be omitted from acceptance calculations if it is identical between states.

Define:

\[
H(\theta,p)=U(\theta)+K(p).
\]

The joint target is:

\[
p(\theta,p)\propto e^{-H(\theta,p)}.
\]

At each HMC iteration:

1. sample fresh momentum \(p\sim N(0,M)\);
2. integrate Hamiltonian dynamics;
3. compute change in Hamiltonian;
4. accept/reject using a Metropolis correction.

---

# 7. Leapfrog integrator

Use the standard velocity/momentum Verlet (leapfrog) integrator.

Given step size epsilon:

### Half momentum update

\[
p_{t+1/2}=p_t-\frac{\epsilon}{2}\nabla U(\theta_t)
\]

### Full position update

\[
\theta_{t+1}=\theta_t+\epsilon M^{-1}p_{t+1/2}
\]

### Half momentum update

\[
p_{t+1}=p_{t+1/2}-\frac{\epsilon}{2}\nabla U(\theta_{t+1}).
\]

Implement this as a pure function:

```python
def leapfrog(
    theta,
    momentum,
    step_size,
    n_steps,
    potential_fn,
    grad_potential_fn,
    inverse_mass_matrix,
):
    ...
```

For production code, also provide a one-step primitive because NUTS needs repeated leapfrog transitions.

---

# 8. HMC Metropolis acceptance

Let:

\[
H_0=H(\theta_0,p_0)
\]

and:

\[
H_1=H(\theta_1,p_1).
\]

The log acceptance probability is:

\[
\log\alpha=\min(0,-H_1+H_0).
\]

Accept when:

```python
np.log(rng.uniform()) < min(0.0, -delta_h)
```

where:

```python
delta_h = H_new - H_old
```

Always perform this correction even though leapfrog is approximately volume preserving and approximately energy conserving. The correction compensates for numerical integration error.

---

# 9. Numerical stability for HMC

Never compute probabilities directly.

Use:

```python
log_alpha = min(0.0, -delta_h)
```

and compare in log space.

Detect pathological Hamiltonian values:

```python
if not np.isfinite(H_new):
    reject
    mark divergent_or_invalid = True
```

Do not silently convert NaNs to zeros or continue.

Define a configurable threshold for an extreme energy error, for example:

```python
energy_error_threshold = 1000.0
```

but treat this as a diagnostic threshold rather than changing the target distribution.

Record:

- `energy_error`;
- `max_energy_error`;
- divergence flag;
- numerical failure flag.

---

# 10. Mass matrix / metric

The mass matrix controls the momentum distribution:

\[
p\sim N(0,M)
\]

and therefore affects how HMC traverses parameter space.

A poorly scaled parameterization can make HMC inefficient.

Support at least:

### Diagonal metric

\[
M=\operatorname{diag}(m_1,\ldots,m_d).
\]

This should be the default production option because it is robust and cheap.

### Dense metric

\[
M\succ0.
\]

Dense adaptation can help strongly correlated posteriors but costs:

- \(O(d^2)\) storage;
- matrix factorization cost;
- matrix-vector multiplication cost.

For the current spectral problem \(d=3K\) is small enough that dense metrics are feasible, but start with diagonal and make dense an explicit configuration option.

---

# 11. Momentum generation

For diagonal mass:

\[
p_i=\sqrt{m_i}z_i,
\qquad z_i\sim N(0,1).
\]

For dense mass, use a Cholesky factor:

\[
M=LL^T
\]

and:

\[
p=Lz.
\]

Do not repeatedly invert the mass matrix. Store the inverse or a factorization appropriate for the numerical kernel.

---

# 12. Fixed-length HMC configuration

Provide:

```python
@dataclass(frozen=True)
class HMCConfig:
    num_steps: int
    step_size: float
    target_accept: float = 0.8
    max_energy_error: float = 1000.0
```

Validate:

- `num_steps >= 1`;
- `step_size > 0`;
- `0 < target_accept < 1`.

The number of leapfrog steps \(L\) and step size epsilon should be considered adaptation parameters unless explicitly fixed by the caller.

---

# 13. Dual averaging for step-size adaptation

Production HMC/NUTS should not rely on a hard-coded step size.

Use dual averaging during warmup.

The target is an average acceptance probability \(\delta\), where typical choices are around 0.8–0.9, with higher values sometimes useful for difficult geometries at the expense of more computation.

A standard dual-averaging scheme uses:

\[
H_t=
\left(1-\frac{1}{t+t_0}\right)H_{t-1}
+
\frac{1}{t+t_0}(\delta-\alpha_t)
\]

and:

\[
\log \epsilon_t=
\mu-\frac{\sqrt{t}}{\gamma}H_t.
\]

Maintain a smoothed final value:

\[
\log\bar\epsilon_t=
 t^{-\kappa}\log\epsilon_t
 +(1-t^{-\kappa})\log\bar\epsilon_{t-1}.
\]

Parameters commonly used in production implementations include values close to:

```text
gamma  = 0.05
t0     = 10
kappa  = 0.75
```

but expose them as configuration rather than burying them in code.

During warmup, adapt epsilon. At production start, freeze it.

---

# 14. Mass-matrix adaptation

Estimate posterior covariance information during warmup.

For a diagonal metric, estimate marginal variances.

For a dense metric, estimate the covariance matrix.

Use numerically stable online estimators, e.g. Welford-style updates.

Do not calculate covariance by repeatedly stacking every warmup sample if avoidable.

Regularize:

\[
\hat\Sigma_\text{regularized}=
\hat\Sigma+\lambda I
\]

with a small configurable ridge term.

Make sure the resulting metric remains symmetric positive definite.

Warmup should be split into phases so that:

1. early adaptation finds a reasonable step size;
2. covariance/metric estimates are collected in expanding windows;
3. final step-size adaptation occurs after metric updates;
4. the final metric and epsilon are frozen before production sampling.

A robust warmup design should resemble:

```text
initial fast adaptation
        |
        v
slow metric window
        |
        v
larger metric window
        |
        v
final fast step-size adaptation
        |
        v
freeze adaptation
        |
        v
production
```

---

# 15. Initial step-size heuristic

Before dual averaging, find a reasonable initial epsilon using a bounded doubling/halving procedure.

Conceptually:

```text
start with epsilon
      |
simulate one leapfrog trajectory
      |
is acceptance sensible?
   /              \\
too high          too low
  |                 |
increase epsilon   decrease epsilon
  \\                 /
       repeat
```

The exact heuristic should be implemented in a bounded, numerically safe manner.

It should:

- avoid infinite loops;
- enforce minimum and maximum step sizes;
- stop on non-finite Hamiltonians;
- return diagnostics.

---

# 16. NUTS: purpose

NUTS extends HMC by eliminating the need to select a fixed trajectory length \(L\).

HMC requires:

\[
T=L\epsilon.
\]

If \(L\) is too short, the sampler behaves too much like a random walk.

If \(L\) is too long, computation is wasted and trajectories can double back.

NUTS adaptively builds a trajectory and stops when it begins to make a U-turn.

The original NUTS paper describes it as recursively constructing a set of candidate states and stopping when the trajectory begins retracing itself.

---

# 17. NUTS state

A NUTS transition should maintain at least:

```python
@dataclass
class NUTSState:
    theta
    momentum
    potential_energy
    kinetic_energy
    log_joint
```

and tree-building state:

```python
@dataclass
class TreeState:
    theta_minus
    theta_plus
    momentum_minus
    momentum_plus

    theta_proposal

    log_weight
    n_valid

    stop
    continue_probability

    sum_accept_prob
    n_alpha

    depth
    divergent
```

Use immutable/pure structures where possible for JAX compatibility.

---

# 18. NUTS algorithm variant

There are multiple mathematically related NUTS variants.

For production code, choose one formulation explicitly and test it independently.

Prefer a modern multinomial-sampling NUTS formulation if implementing a new sampler because it behaves well in practice and is used by modern frameworks.

Do not mix pieces of different NUTS algorithms without deriving the resulting transition kernel.

The coding agent should document exactly which algorithmic variant is implemented.

---

# 19. NUTS tree-building algorithm

Start at:

\[
\theta_0,\quad p_0.
\]

Draw momentum.

Define a joint log-density threshold, or equivalent variable depending on the chosen NUTS formulation.

Begin with a depth-0 tree.

At each depth:

1. randomly choose direction \(v\in\{-1,+1\}\);
2. integrate the existing trajectory one step/tree in direction \(v\);
3. recursively double the trajectory size;
4. combine the new tree with the old tree;
5. check the U-turn criterion;
6. check divergence/numerical failure;
7. continue until a stop condition is reached or maximum tree depth is hit.

Conceptually:

```text
depth 0:        x

depth 1:       x -> x
             /
            x

depth 2:   x <-> x ----> x ----> x

depth 3:   increasingly long trajectory
             |
             STOP
               ^
          U-turn or max depth
```

---

# 20. U-turn criterion

For a Euclidean metric, use the generalized criterion:

\[
(\theta^+-\theta^-)
\cdot M^{-1}p^-
\le 0
\]

or:

\[
(\theta^+-\theta^-)
\cdot M^{-1}p^+
\le 0.
\]

If either condition is satisfied, stop tree expansion.

Implement this as a dedicated, independently tested function:

```python
def is_uturn(theta_minus, theta_plus,
             momentum_minus, momentum_plus,
             inverse_mass_matrix) -> bool:
    ...
```

Do not bury this mathematical condition inside the recursive tree-building code.

---

# 21. Divergence detection in NUTS

A divergent transition means the numerical trajectory fails to preserve Hamiltonian behavior adequately, typically due to regions of problematic curvature or an overly large step size.

Track:

```text
divergent
energy_error
max_energy_error
```

A divergence is not merely a low acceptance-rate event.

Do not hide divergences by automatically rejecting them without recording them.

Production results with substantial divergences should be considered suspect.

Increasing target acceptance reduces step size and can help, but persistent divergences generally require diagnosing the geometry/parameterization rather than blindly increasing tuning.

---

# 22. Maximum tree depth

NUTS doubles tree size.

At maximum depth \(D\), the trajectory contains on the order of:

\[
2^D
\]

leapfrog steps.

Therefore max tree depth is an important computational safety valve.

Provide:

```python
max_tree_depth: int = 10
```

as a configurable default.

Record:

```text
tree_depth
hit_max_tree_depth
n_leapfrog_steps
```

If many draws repeatedly hit maximum tree depth, report a diagnostic warning. Do not automatically increase the maximum indefinitely.

---

# 23. NUTS adaptation configuration

Recommended interface:

```python
@dataclass(frozen=True)
class NUTSConfig:
    target_accept: float = 0.8
    max_tree_depth: int = 10

    adapt_step_size: bool = True
    adapt_mass_matrix: bool = True
    dense_mass: bool = False

    init_buffer: int = ...
    window_size: int = ...
    term_buffer: int = ...
```

Validate:

- target acceptance strictly between 0 and 1;
- max tree depth >= 1;
- warmup windows are feasible given total warmup;
- mass matrix dimensions match parameter dimension.

---

# 24. Critical distinction: warmup vs sampling

Adaptation changes the transition kernel.

Therefore:

```text
warmup samples
```

are not ordinary posterior draws and must not be treated as such.

After warmup:

- step size is frozen;
- mass matrix is frozen;
- all adaptation state is frozen;
- production samples are generated from the fixed kernel.

Return warmup diagnostics separately.

---

# 25. Multiple chains

Production code should support multiple independent chains.

For chain \(c\), initialize independently.

Do not rely on a single chain.

At minimum support:

```python
num_chains
random_seed
chain_id
```

Use a deterministic random-number-generation scheme so that the same seed/configuration gives reproducible results within the same numerical/backend environment.

---

# 26. Convergence diagnostics

Implement or integrate diagnostics for:

## R-hat

Use rank-normalized split-\(\hat R\) where practical.

Values close to:

\[
\hat R=1
\]

are desired.

Do not treat a single arbitrary threshold as proof of convergence.

## Effective sample size

Report:

- bulk ESS;
- tail ESS.

The useful quantity for performance comparison is approximately:

\[
\text{ESS per second}.
\]

## Autocorrelation

Provide plots or summaries for representative parameters.

## Divergences

Report count and fraction.

## Tree depth

Report fraction of samples hitting maximum depth.

## Energy diagnostics

Track energy error and, where appropriate, BFMI-like diagnostics.

---

# 27. Initialization

Do not always start directly at the prior draw if it produces pathological posterior geometry.

Support at least:

1. prior initialization;
2. user-supplied initial parameters;
3. deterministic MAP/optimization-assisted initialization as an optional utility.

For HMC/NUTS, validate the initial state before warmup:

```text
log density finite?
gradient finite?
energy finite?
constraint transforms valid?
```

If initialization fails, report a precise error rather than silently replacing the state.

---

# 28. Parallel tempering + NUTS

This is the intended integration with the paper.

For inverse temperatures:

\[
0=\beta_1<\beta_2<\cdots<\beta_L=1
\]

maintain one state per temperature.

For each iteration:

```text
for each replica l:
    run one NUTS transition targeting q_beta_l

attempt exchanges between adjacent replicas
```

The exchange probability remains:

\[
\alpha_\text{swap}
=\min\left(
1,
\exp\left[
\frac{n}{\sigma^2}
(\beta_{l+1}-\beta_l)
(E(\theta_{l+1})-E(\theta_l))
\right]
\right).
\]

The inner NUTS kernel does not change this formula.

---

# 29. Warmup strategy for tempered NUTS

Recommended initial production strategy:

## Phase A: independent warmup

For every beta:

```text
NUTS warmup
```

with no swaps.

Learn:

- step size;
- metric.

## Phase B: freeze adaptation

Freeze all NUTS adaptation parameters.

## Phase C: production parallel tempering

Run:

```text
NUTS update for each replica
+
adjacent exchange
```

An advanced implementation may eventually support joint/adaptive tempering, but do not start there.

---

# 30. Why each beta should generally have its own NUTS tuning

The target density is:

\[
q_\beta(\theta)
\propto
\exp\left[-\frac{n}{\sigma^2}\beta E(\theta)\right]\phi(\theta).
\]

Its geometry changes with beta.

At beta near 0:

- likelihood contribution is weak;
- distribution is broad;
- prior dominates.

At beta near 1:

- likelihood contribution is strong;
- posterior may be narrow and highly curved.

Therefore one step size and one metric for all replicas may work poorly.

Use per-replica adaptation initially.

---

# 31. Replica identity vs state identity

A replica has:

```text
beta_l
kernel_l
adaptation_l
```

A state has:

```text
theta
energy
```

When two replicas exchange states:

```text
beta stays with replica
kernel stays with replica
adaptation stays with replica
theta moves between replicas
```

Do not swap sampler configurations.

If replica A at beta=0.5 swaps theta with replica B at beta=1.0:

```text
A.beta remains 0.5
A.theta becomes old B.theta

B.beta remains 1.0
B.theta becomes old A.theta
```

This distinction is essential.

---

# 32. Evidence estimation

The paper uses:

\[
Z(1)=
\prod_l
\frac{Z(\beta_{l+1})}{Z(\beta_l)}
\]

and estimates ratios using samples from:

\[
q_{\beta_l}.
\]

Maintain the existing evidence implementation.

Do not mix up the target used for sampling with the estimator.

For numerical stability, calculate:

\[
\log Z(1)=
\sum_l
\log\left[
\frac{Z(\beta_{l+1})}{Z(\beta_l)}
\right].
\]

Each ratio should be estimated in log space using `logsumexp` / `logmeanexp` where appropriate.

---

# 33. Evidence and NUTS diagnostics

Evidence estimates are often more sensitive than ordinary posterior means.

Therefore record:

- effective samples at each beta;
- exchange acceptance by adjacent pair;
- variance of each log-ratio estimator;
- uncertainty estimate for log evidence;
- sensitivity to temperature-ladder spacing.

Do not declare evidence convergence merely because the beta=1 chain has good R-hat.

The whole beta path matters.

---

# 34. Temperature ladder

Treat the beta ladder as an explicit object:

```python
@dataclass(frozen=True)
class TemperatureLadder:
    beta: Array

    def validate(self):
        ...
```

Required invariants:

```text
0 <= beta[0]
beta[0] == 0
beta is strictly increasing
beta[-1] == 1
```

Support the paper's geometric schedule.

Also provide a convenience function for constructing alternative ladders.

For production use, consider diagnostics for adjacent swap rates.

A very low swap rate can indicate that neighboring temperatures are too far apart.

A very high rate is not necessarily bad, but may mean the ladder is using more replicas than needed.

---

# 35. Vectorization and compilation

For performance, the model evaluation should be vectorizable across observations.

Prefer:

```text
theta -> basis parameters -> prediction vector -> residual vector -> log density
```

using array operations.

For multiple replicas, consider batching:

```python
theta.shape == (num_replicas, dimension)
```

where the backend permits it.

JAX is strongly recommended for the HMC/NUTS implementation because:

- gradients can be obtained by automatic differentiation;
- numerical kernels can be JIT compiled;
- array operations can be batched;
- CPU/GPU execution can be supported.

Keep a small NumPy reference implementation for tests and debugging.

---

# 36. JAX requirements

Write pure functions.

Avoid inside model functions:

```python
list.append(...)
random state mutation
global mutable state
Python-side logging
```

Prefer explicit PRNG keys.

For example:

```python
key, subkey = jax.random.split(key)
```

Do not reuse the same key for independent random draws.

Use 64-bit floating point for scientific sampling if available and appropriate:

```python
jax.config.update("jax_enable_x64", True)
```

Precision choice should be documented because posterior geometry and energy errors can be sensitive to floating-point precision.

---

# 37. Automatic differentiation tests

Before trusting HMC/NUTS, verify gradients.

For a random valid point theta:

1. compute automatic gradient;
2. compute finite-difference gradient;
3. compare them.

Use a relative-error test such as:

\[
\frac{|g_\text{AD}-g_\text{FD}|}
{\max(1,|g_\text{AD}|,|g_\text{FD}|)}
<\text{tolerance}.
\]

Use central finite differences for testing only.

Test several beta values, random parameter points, model sizes K, basis types, and prior types.

Exclude points extremely near singularities or constraints unless the expected behavior is explicitly tested.

---

# 38. Integrator correctness tests

For a simple known potential such as a standard normal:

\[
U(q)=\frac12 q^2
\]

use this as the first HMC test.

Test:

- leapfrog roughly conserves Hamiltonian for small epsilon;
- reversibility;
- position/momentum update correctness;
- accepted samples approximate \(N(0,1)\).

Reversibility test:

```text
(theta, p)
    |
forward L steps
    |
(theta', p')
    |
flip momentum
    |
reverse L steps
    |
(theta, -p)
```

up to floating-point error.

This is one of the most valuable low-level tests.

---

# 39. Statistical HMC tests

Use simple distributions with known answers.

### Standard normal

\[
\theta\sim N(0,I).
\]

Check mean, variance, R-hat, and divergence behavior.

### Correlated Gaussian

\[
\theta\sim N(0,\Sigma)
\]

with strong correlation.

Compare:

- random-walk Metropolis;
- diagonal-metric HMC;
- dense-metric HMC.

The goal is to demonstrate that metric adaptation materially improves sampling.

### Banana-shaped distribution

Use a curved target to test geometry.

### Multimodal mixture

Use this to show a key limitation:

- HMC is good within a mode;
- ordinary HMC may not move between well-separated modes;
- tempered HMC/NUTS should improve mode crossing.

---

# 40. NUTS-specific unit tests

Test the tree builder independently.

Required tests include:

1. depth 0 creates a valid one-step tree;
2. forward integration updates only the plus endpoint;
3. backward integration updates only the minus endpoint;
4. tree combining preserves valid endpoint bookkeeping;
5. U-turn detection works in obvious synthetic cases;
6. divergence detection triggers for intentionally unstable epsilon;
7. maximum tree depth terminates;
8. no NaNs are returned;
9. multinomial candidate selection is statistically sensible;
10. NUTS samples a known Gaussian correctly.

Avoid only testing end-to-end output. Bugs in tree recursion can remain hidden if the target is easy.

---

# 41. Reproducibility tests

Given the same model, data, seed, and configuration, tests should be reproducible within the same numerical/backend environment.

Test separate chains use independent RNG streams.

For JAX, derive keys hierarchically:

```text
master key
   ├── chain 0
   │    ├── warmup
   │    └── production
   ├── chain 1
   ...
```

For replicas:

```text
chain
   ├── replica 0
   ├── replica 1
   ...
```

Avoid sharing the same key between independent stochastic operations.

---

# 42. Production result object

Return a structured result.

For example:

```python
@dataclass
class MCMCResult:
    samples: Array
    warmup_diagnostics: Diagnostics
    sampling_diagnostics: Diagnostics

    step_size: Array
    mass_matrix: Array

    acceptance_rate: float
    divergences: int

    tree_depth: Array
    num_leapfrog_steps: Array

    energy: Array
```

For tempered sampling:

```python
@dataclass
class TemperedMCMCResult:
    beta: Array
    states: Array
    energies: Array

    swap_acceptance: Array
    replica_traces: Array

    sampler_results: list[MCMCResult]
```

For model selection:

```python
@dataclass
class ModelEvidence:
    K: int
    log_evidence: float
    uncertainty: float
```

Avoid returning a huge raw tuple whose fields depend on execution mode.

---

# 43. Diagnostics API

Provide functions approximately like:

```python
summarize(result)
plot_trace(result)
plot_energy(result)
plot_autocorrelation(result)
plot_rank(result)
plot_temperature_traces(result)
plot_swap_acceptance(result)
check_convergence(result)
```

Do not make plotting functions necessary for inference.

Diagnostics should also be usable in headless CI environments.

---

# 44. Failure handling

Raise errors for configuration problems:

```text
negative step size
invalid target_accept
malformed beta ladder
dimension mismatch
non-SPD mass matrix
invalid initial state
```

For sampling-path numerical failures, record diagnostics rather than necessarily crashing the entire run.

However, repeated failures should trigger an explicit warning or configurable failure policy.

Never silently:

- drop NaN samples;
- replace invalid gradients with zeros;
- clip the energy without recording it;
- ignore divergent transitions.

---

# 45. Performance targets

Benchmark the following independently:

1. forward model;
2. log density;
3. gradient;
4. leapfrog step;
5. one HMC transition;
6. one NUTS transition;
7. one full tempered iteration.

Report:

```text
seconds / transition
gradient evaluations / second
effective samples / second
ESS / gradient evaluation
```

Do not optimize solely for transitions/second.

The main scientific metric is effective sampling efficiency.

---

# 46. Caching

Within one transition, cache:

```text
theta
log_density(theta)
potential_energy(theta)
gradient(theta)
```

Do not recompute the gradient for an unchanged state.

For the spectral model also consider caching:

```text
x-grid
data
noise variance
constant likelihood terms
prior hyperparameters
```

Avoid caching mutable sampler state inside the model.

---

# 47. Basis-function requirements for HMC/NUTS

Every basis implementation used with HMC/NUTS should document:

```python
class BasisFunction:
    parameter_dimension: int

    def evaluate(self, x, parameters):
        ...

    def validate_parameters(self, parameters):
        ...
```

The basis must be differentiable with respect to sampled parameters except at explicitly unsupported singularities.

Document:

- support/domain;
- parameter constraints;
- parameter interpretation;
- derivatives/AD compatibility;
- numerical stability.

For example, a Lorentzian basis:

\[
\phi(x;\mu,\gamma)=
\frac{1}{1+((x-\mu)/\gamma)^2}
\]

requires:

\[
\gamma>0.
\]

The preferred HMC implementation should sample an unconstrained transform of gamma.

---

# 48. Prior requirements

Each prior should own both sampling and density evaluation.

Example interface:

```python
class Prior(Protocol):
    def sample(self, key, shape) -> Array:
        ...

    def log_prob(self, value) -> Scalar:
        ...
```

If a constrained transform is used, clearly distinguish:

```text
Prior in constrained space
        +
Jacobian
        ↓
target in unconstrained space
```

from:

```text
Prior directly defined in unconstrained space
```

Do not accidentally apply a Jacobian twice.

Useful implementations:

```text
GammaPrior
NormalPrior
LogNormalPrior
UniformPrior
TruncatedNormalPrior
JointPrior
IndependentPrior
```

Add explicit tests for support and normalization where analytically feasible.

---

# 49. HMC vs NUTS API

Expose both:

```python
sample_hmc(...)
sample_nuts(...)
```

behind a common sampler interface:

```python
sampler.sample(
    initial_state,
    num_warmup,
    num_samples,
    rng_key,
    target,
)
```

This lets parallel tempering select:

```python
inner_sampler="hmc"
```

or:

```python
inner_sampler="nuts"
```

without changing the exchange implementation.

---

# 50. Recommended implementation sequence

Do NOT implement the complete system in one pass.

Use this order:

## Phase 1 — differentiable target

Implement model, prior, transform, log_density, and gradient. Verify gradients.

## Phase 2 — leapfrog

Implement fixed-step leapfrog. Test on a standard normal.

## Phase 3 — fixed HMC

Implement momentum draw, leapfrog, Hamiltonian, and Metropolis correction. Test distributional correctness.

## Phase 4 — step-size adaptation

Implement initial epsilon finder and dual averaging. Test on several target scales.

## Phase 5 — mass adaptation

Implement diagonal metric first, then dense metric.

## Phase 6 — NUTS

Implement tree building, U-turn criterion, candidate selection, divergence handling, and max depth. Test against known distributions.

## Phase 7 — multi-chain diagnostics

Add R-hat, ESS, energy diagnostics, and divergence summaries.

## Phase 8 — parallel tempering

Replace the current inner Metropolis kernel with NUTS. Keep exchange acceptance unchanged.

## Phase 9 — evidence

Verify evidence estimates agree between Metropolis+exchange, HMC+exchange, and NUTS+exchange within Monte Carlo uncertainty.

## Phase 10 — performance optimization

Only after correctness: JAX, jit, vmap, 64-bit, batching, and profiling.

---

# 51. Reference implementation strategy

Maintain a deliberately slow reference implementation.

For example:

```text
reference_numpy/
    pure Python/NumPy HMC
```

and:

```text
production_jax/
    JAX HMC/NUTS
```

Tests should compare:

- log density;
- gradient;
- leapfrog trajectory;
- Hamiltonian error.

The production implementation should not be trusted merely because it is fast.

---

# 52. Acceptance-rate guidance

Acceptance rate is a diagnostic, not a goal by itself.

A low acceptance rate can indicate:

- step size too large;
- poor geometry;
- numerical instability.

A very high acceptance rate can indicate:

- step size unnecessarily small;
- overly conservative computation.

Do not write code that continually changes parameters during production solely to chase a target acceptance rate.

Adapt during warmup; freeze afterward.

---

# 53. Common failure modes and remedies

## Divergences

Possible causes:

- posterior geometry with very high curvature;
- poor parameterization;
- overly large step size;
- strong nonlinear parameter correlations.

Try, in order:

1. inspect divergent locations;
2. improve parameterization;
3. increase target acceptance / reduce step size;
4. reconsider priors;
5. consider a better mass matrix.

Do not simply suppress the warning.

## Maximum tree depth

Possible causes:

- long autocorrelation;
- poor geometry;
- overly conservative step size.

Diagnose before increasing the limit.

## Very low ESS

Possible causes:

- high posterior correlation;
- poor metric;
- insufficient warmup;
- multimodality.

Tempering helps multimodality; metric adaptation helps correlation.

## NaNs

Investigate overflow/underflow, invalid transformed parameters, unstable basis functions, invalid prior domains, and bad initial values.

---

# 54. Special concern for the spectral model

The mixture model can have difficult geometry because:

- amplitude and width can trade off;
- neighboring peak centers can strongly interact;
- multiple peaks can become nearly redundant;
- label switching produces equivalent posterior modes;
- unnecessary components can create weakly identified parameters.

These are exactly the conditions under which diagnostics matter more than simply getting an answer.

For candidate models with large K, inspect:

```text
posterior geometry
parameter correlations
divergences
ESS
swap rates
evidence uncertainty
```

Do not assume that NUTS automatically solves all problems.

---

# 55. HMC/NUTS does not replace tempering

This should remain explicit in the code documentation.

HMC/NUTS is excellent at exploring a connected high-probability region.

It does not guarantee efficient transitions between strongly separated modes.

The architecture should therefore remain:

```text
NUTS within temperature
+
exchange across temperatures
```

rather than:

```text
NUTS instead of exchange Monte Carlo
```

for the paper's multimodal spectral-deconvolution problem.

---

# 56. Testing the combined tempered NUTS sampler

Use a synthetic multimodal target first.

Example:

\[
p(\theta)=
0.5N(-4,1)+0.5N(4,1).
\]

Run:

### Case A

One-temperature NUTS.

Expected behavior:

- good within-mode sampling;
- possible poor cross-mode mixing.

### Case B

Tempered NUTS.

Expected behavior:

- hot chains explore both modes;
- states exchange;
- cold chain receives states from multiple modes more often.

Record:

- mode occupancy;
- number of mode switches;
- replica temperature trajectories.

This is more informative than testing only the spectral model.

---

# 57. Evidence regression test

Create a small problem where the target integral can be computed numerically using quadrature or a known analytic result.

Compare:

```text
exact log Z
Monte Carlo log Z
tempered HMC log Z
tempered NUTS log Z
```

within expected Monte Carlo uncertainty.

This test is essential because a sampler can look excellent while still giving a poor evidence estimate.

---

# 58. Model-selection regression test

Use the paper's synthetic setup with known:

\[
K_\text{true}=3.
\]

The exact paper used 301 points, noise variance 0.01, and candidate models \(K=1,\ldots,8\).

For an initial CI test, use a smaller computational budget.

Check that the selected K is usually 3.

Do not assert that a short educational run must reproduce the paper's reported 93/100 frequency.

For a serious benchmark, reproduce the paper's repeated-run setup with a documented computational budget and compare distributions rather than one run.

---

# 59. Benchmark matrix

Create a benchmark table:

| Sampler | Metric | Tempering | ESS/sec | Divergences | Swap rate | log-Z error |
|---|---|---|---:|---:|---:|---:|
| Metropolis | n/a | yes | | | | |
| HMC | diagonal | yes | | | | |
| HMC | dense | yes | | | | |
| NUTS | diagonal | yes | | | | |
| NUTS | dense | yes | | | | |

The purpose is not to assume NUTS wins every metric.

The benchmark should reveal where the computational benefit actually comes from.

---

# 60. Production configuration example

Provide a configuration resembling:

```python
config = TemperedNUTSConfig(
    num_chains=4,
    num_warmup=2000,
    num_samples=2000,

    nuts=NUTSConfig(
        target_accept=0.9,
        max_tree_depth=10,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
    ),

    temperature=TemperatureConfig(
        num_replicas=24,
        schedule="geometric",
    ),

    evidence=EvidenceConfig(
        estimator="log_ratio",
    ),
)
```

The exact defaults should be validated empirically rather than copied blindly from another framework.

---

# 61. Logging and observability

Production runs should report periodically:

```text
chain
iteration
warmup/sample phase
mean acceptance probability
step size
average tree depth
mean leapfrog steps
divergence count
max energy error
ESS if available
swap acceptance by pair
```

Do not print every iteration by default.

Provide a verbosity setting:

```text
0 = silent
1 = summary
2 = periodic diagnostics
3 = detailed debugging
```

---

# 62. Serialization

Save:

- configuration;
- random seed;
- model specification;
- prior specification;
- temperature ladder;
- final adaptation state;
- posterior samples;
- diagnostics.

A production run should be reconstructable.

Do not rely on pickling arbitrary Python objects as the only reproducibility mechanism.

Prefer explicit metadata formats such as JSON/YAML for configuration and a documented binary array format for samples.

---

# 63. API stability

Avoid exposing internal tree-builder classes as the main public interface.

Public API should be small:

```python
fit_hmc(...)
fit_nuts(...)
fit_tempered_nuts(...)
estimate_log_evidence(...)
check_convergence(...)
```

Everything else can remain internal until stable.

---

# 64. Documentation requirements

The README should contain:

1. a five-minute example;
2. a mathematical model description;
3. a sampler-selection guide;
4. diagnostics explanation;
5. reproducibility instructions;
6. benchmark instructions.

The notebook should explain the algorithm visually, but should use the same public APIs as production code.

No notebook-only algorithm implementations should be used for the actual benchmark.

---

# 65. External algorithm references

Use the following primary references when implementing and reviewing the sampler:

- Hoffman & Gelman, **The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo** (2014): foundational NUTS algorithm and dual-averaging step-size adaptation.
- Betancourt, **A Conceptual Introduction to Hamiltonian Monte Carlo** (2017): geometry, diagnostics, and practical failure modes.
- Betancourt et al., **The Geometric Foundations of Hamiltonian Monte Carlo** (2017): geometric foundations.
- Current **Stan Reference Manual**, HMC/NUTS chapters: production implementation details including warmup, metric adaptation, and divergent transitions.
- Current **PyMC documentation** for practical NUTS configuration and diagnostics.
- Current **NumPyro documentation** for JAX-based NUTS implementation concepts.

Use the current version of official documentation during implementation because APIs and defaults change over time.

---

# 66. Acceptance criteria

The implementation is ready for production review only when all of the following are true.

## Mathematical correctness

- [ ] HMC samples known Gaussian targets correctly.
- [ ] Leapfrog is numerically reversible within tolerance.
- [ ] Hamiltonian error behaves as expected as epsilon decreases.
- [ ] Metropolis correction is correct.
- [ ] NUTS satisfies the intended transition algorithm.
- [ ] U-turn criterion is unit tested.
- [ ] warmup adaptation freezes before production.
- [ ] transformed constrained parameters include the correct Jacobian.
- [ ] exchange acceptance matches the paper's formula.

## Statistical correctness

- [ ] Multiple chains are supported.
- [ ] R-hat is computed.
- [ ] bulk/tail ESS are computed or delegated to a tested diagnostics package.
- [ ] divergences are recorded and surfaced.
- [ ] maximum-tree-depth events are recorded.
- [ ] energy diagnostics are available.
- [ ] evidence estimates have uncertainty diagnostics.
- [ ] model selection is tested on synthetic data.

## Engineering correctness

- [ ] no global mutable sampler state;
- [ ] deterministic seeded tests;
- [ ] no silent NaN handling;
- [ ] configuration validation;
- [ ] type annotations;
- [ ] unit tests;
- [ ] integration tests;
- [ ] performance benchmarks;
- [ ] documentation;
- [ ] reproducible configuration and metadata.

## Performance

- [ ] log-density is vectorized;
- [ ] gradients are compiled/autodiff;
- [ ] repeated calculations are cached where valid;
- [ ] production sampler does not execute unnecessary Python-level loops in the numerical hot path;
- [ ] ESS/sec is benchmarked.

---

# 67. Recommended technology choice

For this project, the recommended production stack is:

```text
Python
  +
JAX
  +
mature NUTS implementation OR a carefully tested local implementation
  +
custom parallel-tempering orchestration
  +
ArviZ-style diagnostics
```

A mature NUTS implementation should normally be preferred to maintaining a home-grown implementation unless the purpose is algorithm research.

A hand-written HMC implementation is still valuable as:

- a reference implementation;
- a teaching tool;
- a test oracle for basic kernels.

Do not treat a teaching implementation as production-ready simply because it passes a few end-to-end tests.

---

# 68. Final architecture

The final system should look like:

```text
                       Scientific model
                              |
               +--------------+--------------+
               |                             |
             Basis                         Prior
               |                             |
               +--------------+--------------+
                              |
                         log density
                              |
                         transforms
                              |
                    unconstrained target
                              |
                    automatic differentiation
                              |
              +---------------+---------------+
              |                               |
             HMC                             NUTS
              |                               |
              +---------------+---------------+
                              |
                      per-beta kernels
                              |
                     parallel tempering
                              |
                 +------------+-------------+
                 |                          |
             posterior                  beta samples
              beta=1                         |
                 |                           |
                 +-------------+-------------+
                               |
                         evidence estimator
                               |
                         model comparison
                               |
                              best K
```

The key engineering principle is:

> HMC/NUTS should be a replaceable inference engine operating on a generic differentiable log-density, not a spectral-deconvolution-specific implementation.

That design will also make future basis functions and priors cheap to introduce, which is the motivation for the preceding refactor.
