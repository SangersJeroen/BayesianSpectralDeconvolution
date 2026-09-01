# Bayesian Spectral Deconvolution — Implementation & Refactor Summary

## Purpose

Refactor the current educational implementation of Bayesian spectral deconvolution into a modular, extensible codebase that cleanly separates:

- basis-function definitions;
- parameterization/constraints;
- prior distributions;
- likelihood / observation model;
- Bayesian target construction;
- within-temperature MCMC kernels;
- exchange Monte Carlo / parallel tempering;
- marginal-likelihood (evidence) estimation;
- model selection over model size `K`;
- diagnostics, plotting, examples, and tests.

The primary design goal is that changing a basis function or changing a prior should require changing only the corresponding model component, rather than editing sampler or model-selection code.

The original paper uses the generic RBF form

    f(x; theta) = sum_k a_k phi_k(x)

and specializes `phi_k` to a Gaussian basis. Its Bayesian posterior, tempered target, exchange probability, and evidence calculation are written in terms of the resulting model/error and therefore do not inherently depend on the Gaussian choice. See the paper's Eqs. (1)–(2), (7), (10)–(15).

---

## Current implementation

The existing reference implementation is:

- `/mnt/data/bayesian_spectral_deconvolution.py`

It currently contains, in one module:

- Gaussian basis evaluation;
- RBF spectrum prediction;
- paper-style mean-squared error;
- synthetic and MGM prior parameter containers;
- scalar log-PDF helper functions;
- `log_prior_theta`;
- Bayesian target calculation;
- paper temperature ladder;
- random-walk Metropolis;
- exchange Monte Carlo;
- evidence estimation;
- model selection over `K`;
- synthetic-data generation.

Treat this implementation as the behavioral reference during refactoring. Preserve numerical behavior first; optimize afterward.

---

# 1. Design principles

## 1.1 Separation of concerns

A scientific model should not know how it is sampled.

A sampler should not know what a Gaussian/Lorentzian/Voigt basis is.

A prior should not know how the likelihood is evaluated.

Evidence estimation should consume samples/energies and not depend directly on the sampler implementation.

## 1.2 Prefer composition over special cases

Do not create functions such as:

- `log_gaussian_rbf_prior_theta`
- `log_lorentzian_rbf_prior_theta`
- `log_gaussian_mgm_prior_theta`

Instead compose generic components:

    Basis + Parameterization + Prior + Likelihood -> Bayesian model

## 1.3 Keep the paper-faithful implementation available

Do not silently alter the statistical model while refactoring. In particular:

- preserve the paper's parameterization where compatibility is intended;
- preserve the distinction between precision-like `b` and conventional standard deviation;
- preserve the paper's factor of `n / sigma2` in the tempered posterior;
- preserve the exchange acceptance formula;
- preserve the evidence estimator used by the current implementation.

If a performance or parameterization improvement changes the mathematical model, expose it as an explicit alternative rather than replacing the reference behavior.

## 1.4 Make stochastic behavior reproducible

Pass an explicit `numpy.random.Generator` (or a reproducible seed factory) into stochastic components.

Avoid hidden global RNG state.

---

# 2. Proposed package structure

```text
bayesian_spectral/
├── pyproject.toml
├── README.md
├── src/
│   └── bayesian_spectral/
│       ├── __init__.py
│       ├── basis.py
│       ├── parameters.py
│       ├── priors.py
│       ├── likelihoods.py
│       ├── models.py
│       ├── samplers/
│       │   ├── __init__.py
│       │   ├── metropolis.py
│       │   ├── hmc.py
│       │   └── nuts.py
│       ├── tempering.py
│       ├── evidence.py
│       ├── model_selection.py
│       ├── diagnostics.py
│       ├── data.py
│       └── plotting.py
├── tests/
│   ├── test_basis.py
│   ├── test_parameters.py
│   ├── test_priors.py
│   ├── test_likelihoods.py
│   ├── test_models.py
│   ├── test_metropolis.py
│   ├── test_tempering.py
│   ├── test_evidence.py
│   └── test_model_selection.py
└── notebooks/
    └── paper_walkthrough.ipynb
```

A small package is preferable to a single large script once the implementation supports multiple bases, priors, and samplers.

---

# 3. Basis-function abstraction

## 3.1 Generic interface

Create a `BasisFunction` abstraction with at least:

```python
class BasisFunction(Protocol):
    name: str
    n_parameters_per_basis: int

    def evaluate(self, x: Array, params: Array) -> Array:
        ...
```

The exact protocol/class choice is up to the coding agent, but the public behavior should be explicit.

The basis must define one-component evaluation, not the whole `K`-component model.

## 3.2 Gaussian basis

Implement the current paper basis:

    phi(x; mu, b) = exp(-0.5 * b * (x - mu)**2)

with the separate amplitude handled by the model or by a parameterization layer.

For compatibility, keep the current parameter ordering available:

    theta = [a_1...a_K, mu_1...mu_K, b_1...b_K]

## 3.3 Example alternative bases

Add at least one test/example implementation, preferably Lorentzian:

    phi(x; mu, gamma) = 1 / (1 + ((x - mu) / gamma)**2)

This is mainly to prove the architecture actually permits changing basis without touching tempering/evidence/model-selection code.

Potential future bases:

- Voigt;
- pseudo-Voigt;
- Laplace;
- skewed Gaussian;
- user-supplied callable basis.

## 3.4 Vectorization requirement

Basis evaluation must support arrays and avoid Python loops over observations where practical.

Prefer broadcasting shapes such as:

    (K, n)

for `K` components and `n` observations.

---

# 4. Parameterization abstraction

The prior should not be forced to understand how a flattened `theta` vector is laid out.

Introduce a parameterization object responsible for:

- packing structured parameters into a flat vector;
- unpacking a flat vector;
- validation;
- optional unconstrained <-> constrained transforms.

Example conceptual API:

```python
class Parameterization:
    ndim: int

    def pack(self, structured) -> Array: ...
    def unpack(self, theta: Array): ...
    def validate(self, theta: Array) -> None: ...
```

For the current Gaussian model, the structured representation could be:

```text
a:  (K,)
mu: (K,)
b:  (K,)
```

A future basis can use a different number or type of per-component parameters without changing prior/sampler code.

For HMC/NUTS, provide optional transforms for constrained parameters, e.g. positive scales represented internally on the real line.

---

# 5. Prior abstraction — central requested refactor

## 5.1 Goal

Move **all prior generation and prior probability evaluation** into prior objects.

This means the Bayesian model must not contain code such as:

```python
for ak, muk, bk in zip(...):
    ...
```

and must not know that `a` uses Gamma, `mu` uses Normal, etc.

Instead the model calls something like:

```python
log_prior = prior.log_prob(parameters)
```

and initialization calls:

```python
parameters = prior.sample(rng, K)
```

## 5.2 Generic prior interface

Recommended conceptual interface:

```python
class Prior(Protocol):
    def sample(self, rng, parameterization) -> Array:
        ...

    def log_prob(self, theta: Array, parameterization) -> float:
        ...
```

Optionally add:

```python
def validate(self, theta): ...
def metadata(self): ...
```

Do not require a prior to own the `K` concept unless necessary. Prefer a prior object that can be instantiated/configured with `K` through a factory or model configuration.

## 5.3 Composable priors

The preferred design is a product/factorized prior composed from parameter-group priors.

For example:

```text
ModelPrior
├── amplitudes -> GammaPrior
├── centers    -> NormalPrior
└── bandwidths -> GammaPrior
```

Then:

    log p(theta) = sum_j log p_j(theta_j)

For the synthetic paper model this reproduces Eqs. (17)–(19).

## 5.4 Distribution classes

Implement reusable distribution-level classes/helpers, e.g.:

- `GammaPrior(shape, rate)`
- `NormalPrior(mean, precision)`
- `UniformPrior(lower, upper)`
- `TransformedPrior(...)` for positive/constrained parameters where useful
- `IndependentProductPrior([...])`

Each distribution object should expose:

```python
sample(rng, size=None)
log_prob(x)
```

Use vectorized implementations.

## 5.5 Sign-constrained paper MGM priors

Support the paper's olivine-style construction where a negative parameter receives a Gamma prior through a sign transform, e.g.:

    -a ~ Gamma(...)

Do not implement this as a one-off branch in `log_prior_theta`. Use a generic transformed prior or signed-Gamma prior wrapper.

The paper's MGM section defines negative `a_k` and `c_1` using Gamma densities over their negated values.

## 5.6 Important normalization requirement

All priors used for evidence/model selection must be **proper normalized probability densities** unless there is a deliberate, documented reason otherwise.

Do not accidentally omit normalizing constants in `log_prob` when those constants differ between candidate models or prior specifications. They can matter for evidence comparisons.

---

# 6. Likelihood abstraction

Create a likelihood/observation-model abstraction such as:

```python
class Likelihood(Protocol):
    def log_prob(self, y, prediction, context=None) -> float:
        ...
```

Implement the current Gaussian-noise likelihood first.

For the paper's setup:

    y_i = f(x_i; theta) + epsilon_i
    epsilon_i ~ Normal(0, sigma2)

The existing implementation intentionally omits the Gaussian normalizing constant in the target because it is constant in `theta`; preserve that behavior in the internal tempered target where appropriate, but provide a fully normalized likelihood method if useful for diagnostics or future evidence calculations.

The likelihood must remain independent of the basis family.

---

# 7. Bayesian model abstraction

Create a `BayesianModel` that combines:

```text
forward model
likelihood
prior
```

Conceptual API:

```python
class BayesianModel:
    def predict(self, theta): ...
    def log_likelihood(self, theta): ...
    def log_prior(self, theta): ...
    def log_posterior(self, theta): ...
    def log_tempered_target(self, theta, beta): ...
    def energy(self, theta): ...
```

For the paper-faithful Gaussian-noise model:

    E(theta) = 1/(2n) * sum_i (y_i - f(x_i; theta))**2

    log q_beta(theta)
        = -(n / sigma2) * beta * E(theta)
          + log prior(theta)

up to beta-dependent normalization.

The important result is that `BayesianModel` does not contain a Gaussian-specific basis formula.

---

# 8. Sampler abstraction

Create a generic interface such as:

```python
class MCMCKernel:
    def step(self, state, target) -> state:
        ...
```

or an equivalent stateless kernel design.

The tempering layer should accept a kernel factory rather than directly constructing `RandomWalkMetropolis`.

Example:

```python
kernel_factory = lambda beta, rng: RandomWalkMetropolis(...)

# future
kernel_factory = lambda beta, rng: HMC(...)
# or
kernel_factory = lambda beta, rng: NUTS(...)
```

This is important for the later HMC/NUTS modernization.

---

# 9. Refactor exchange Monte Carlo / parallel tempering

The `ExchangeMonteCarlo` component should know only about:

- a target/model;
- the beta ladder;
- per-temperature kernels;
- replica states;
- exchange acceptance.

It should not know about Gaussian priors or Gaussian basis functions.

## 9.1 Replica state

Use a state object containing at least:

```python
theta
log_target
energy
```

Optionally cache:

```python
log_prior
log_likelihood
prediction
```

where doing so materially reduces repeated work.

## 9.2 Exchange rule

Preserve the paper's exchange rule:

    log_v = (n / sigma2) * (beta2 - beta1) * (E2 - E1)

    accept with probability min(1, exp(log_v))

Do calculations in log space.

When swapping, swap states, not temperatures.

## 9.3 Alternating swap parity

Retain the current alternating adjacent-pair schedule:

- even pairs on one exchange round;
- odd pairs on the next.

Make this strategy configurable.

---

# 10. Evidence / marginal likelihood

Move evidence estimation into its own module.

The paper defines:

    z(beta) = integral exp[-(n/sigma2) beta E(theta)] prior(theta) dtheta

with `z(0)=1` and `z(1)=Z(D)`.

Use the paper's ratio identity:

    z(beta[l+1]) / z(beta[l])
      = E_{q_beta[l]}[
            exp(-n/sigma2 * DeltaBeta * E(theta))
        ]

Then:

    log Z = sum_l log ratio_l

The estimator must consume sampled energies and the beta ladder, not a concrete MCMC implementation.

Provide a result object containing at least:

- `log_z`;
- per-interval `log_ratios`;
- uncertainty estimates;
- optionally effective sample size or diagnostics.

Use stable `logsumexp` / `logmeanexp` calculations.

---

# 11. Model selection

`model_selection.py` should:

1. construct a model for each candidate `K`;
2. construct the selected prior for that model;
3. run the tempering sampler;
4. estimate `log Z`;
5. report `-log Z` / stochastic complexity;
6. select the model minimizing stochastic complexity (equivalently maximizing evidence).

The selection layer should not contain basis-specific or prior-specific formulas.

Example conceptual usage:

```python
model = SpectralModel(
    basis=LorentzianBasis(),
    prior=lorentzian_prior,
    likelihood=GaussianNoise(sigma2=...),
)

result = select_model_size(
    model_factory=model_factory,
    K_values=range(1, 9),
    sampler=tempered_sampler,
)
```

---

# 12. Performance refactor

Do the correctness refactor first; then optimize.

Priority order:

## 12.1 Vectorize basis and prediction

Evaluate all `K` components and all observations via NumPy broadcasting.

## 12.2 Cache energy / log density

Avoid recomputing a state's energy when it has not changed.

Avoid recomputing log target during swaps when a decomposition allows a cheap update.

## 12.3 Batch replicas where practical

A later optimized implementation may store all replica parameters as a matrix:

    theta_replicas.shape == (L, ndim)

and evaluate the forward model/energy in batches.

Do not sacrifice clarity in the first refactor if batching makes the code harder to reason about; add it after correctness tests exist.

## 12.4 Avoid unnecessary trace storage

Support configurable recording:

- keep beta=1 samples by default;
- optionally keep all temperatures;
- optionally keep full replica trajectories for diagnostics.

## 12.5 Keep numerical calculations in log space

Particularly for evidence ratios and acceptance probabilities.

---

# 13. HMC/NUTS compatibility

The architecture should make HMC/NUTS possible without a second rewrite.

The model must eventually expose:

```python
log_tempered_target(theta, beta)
```

and, for autodiff-compatible implementations:

```python
grad_log_tempered_target(theta, beta)
```

Do not manually derive/store gradients in the model if using JAX/another autodiff framework unless necessary.

## 13.1 Parameter transforms

HMC/NUTS should work in an unconstrained space where practical.

For positive parameters such as Gaussian `a` and `b`, support transformations such as:

    a = exp(z_a)

or a numerically stable positive transform such as softplus.

If transforming a prior, include the Jacobian in the transformed log density.

Do not use a transformed parameterization in the paper-reference path unless it is explicitly documented, because it can change the effective computational representation and must be handled carefully for evidence.

## 13.2 Tempered NUTS

The tempering layer should be able to construct one kernel per beta:

```text
beta_1 -> NUTS kernel
beta_2 -> NUTS kernel
...
beta_L -> NUTS kernel
```

Each temperature may require its own tuned step size and metric.

Recommended lifecycle:

1. independent warmup/adaptation at each beta;
2. freeze adaptation;
3. begin production sampling;
4. perform adjacent replica exchanges.

The exchange formula itself remains unchanged.

## 13.3 Do not implement production NUTS from scratch initially

A small educational HMC implementation is useful for teaching/tests, but production NUTS should preferably use a mature implementation (e.g. NumPyro/PyMC/JAX ecosystem) once the custom target is validated.

---

# 14. Diagnostics

Add a diagnostics module with:

- within-temperature acceptance rates;
- adjacent exchange acceptance rates;
- energy traces;
- parameter traces;
- beta=1 posterior histograms;
- temperature-space trajectories / replica round trips;
- effective sample size where feasible;
- R-hat for multiple independent chains/runs where applicable.

For HMC/NUTS additionally record:

- divergences;
- step size;
- tree depth;
- energy/BFMI-style diagnostics where supported by the chosen framework.

---

# 15. Testing requirements

The refactor is successful only if the tests demonstrate that changing a basis or prior leaves unrelated inference machinery intact.

## 15.1 Basis tests

For each basis:

- evaluate known values analytically;
- test scalar and vector inputs;
- test broadcasting;
- test parameter validation;
- compare vectorized vs reference loop implementation for Gaussian basis.

## 15.2 Prior tests

For every prior distribution:

- sampled values satisfy support constraints;
- `log_prob` is finite on valid values;
- `log_prob` returns `-inf` outside support where appropriate;
- Monte Carlo moments roughly match known distribution moments;
- normalization can be checked numerically for simple one-dimensional cases;
- transformed priors include Jacobian contributions where applicable.

Important regression test:

```text
Changing SyntheticPrior hyperparameters must change the prior contribution,
but must not require changing the Bayesian model, sampler, or evidence code.
```

## 15.3 Likelihood tests

- compare implementation to direct Gaussian log-likelihood;
- verify energy/log-likelihood relationship used by the paper;
- verify identical behavior regardless of basis implementation.

## 15.4 Exchange tests

- identical energies -> exchange log ratio = 0;
- favorable swap -> probability 1 in the appropriate limiting case;
- verify states swap, not beta values;
- verify exchange acceptance is invariant to irrelevant prior normalizing constants when the same prior is used across replicas.

## 15.5 Evidence tests

Use a toy one-dimensional model with analytically tractable normalization where possible.

Check that:

- `z(0) = 1` for a normalized prior;
- estimated log evidence converges toward a known value;
- log-space and direct implementations agree in numerically safe regimes.

## 15.6 Model-selection regression test

On the paper-like synthetic data, with a modest but reproducible MCMC budget, verify that the code prefers `K=3` often enough to catch regressions. Do not hard-code exact scientific performance from a short run.

---

# 16. Recommended user-facing API

Aim for simple high-level usage.

Example:

```python
model = BayesianSpectralModel(
    x=x,
    y=y,
    basis=GaussianBasis(),
    prior=paper_synthetic_prior(K=3),
    likelihood=GaussianNoise(sigma2=0.01),
)

sampler = ParallelTempering(
    model=model,
    betas=paper_beta_schedule(24),
    kernel_factory=metropolis_kernel_factory,
)

result = sampler.run(
    burn_in=2000,
    samples=2000,
)

evidence = estimate_evidence(model, result)
```

Changing the basis should look like:

```python
basis=LorentzianBasis()
```

Changing the prior should look like:

```python
prior=my_lorentzian_prior(K=3)
```

Changing the sampler should look like:

```python
kernel_factory=nuts_kernel_factory
```

These three changes should be orthogonal.

---

# 17. Migration strategy

Do not rewrite everything at once.

## Phase 1 — Characterization

Freeze the current implementation as a reference.

Add regression tests around:

- `rbf_spectrum`;
- `mean_squared_error`;
- `log_prior_theta`;
- temperature ladder;
- exchange acceptance;
- evidence estimator.

## Phase 2 — Priors

Extract distribution utilities and prior classes.

Replace `log_prior_theta` with a prior object while keeping numerical outputs identical.

This directly addresses the requested refactor.

## Phase 3 — Basis

Extract `GaussianBasis` and refactor prediction to use the basis interface.

Add `LorentzianBasis` as a proof-of-extensibility example.

## Phase 4 — Bayesian model

Separate likelihood, prior, prediction, energy, and tempered target.

## Phase 5 — Tempering

Make parallel tempering accept arbitrary within-temperature kernels.

## Phase 6 — Performance

Vectorize/batch replicas, cache quantities, reduce allocations, and profile before/after.

## Phase 7 — HMC

Add a small educational fixed-step HMC implementation against a toy model.

## Phase 8 — NUTS

Integrate a mature NUTS backend and validate it against the reference Metropolis implementation on small problems.

## Phase 9 — Production diagnostics

Add ESS/R-hat/divergence/round-trip diagnostics and benchmark effective samples/sec.

---

# 18. Important scientific caveats

## 18.1 Evidence is sensitive to the complete model specification

When comparing `K`, the prior is part of the model. Changing the prior can change the evidence materially even when the likelihood is unchanged.

Therefore the refactor must preserve **proper prior normalization** and make prior configuration explicit.

## 18.2 Parameter transformations matter for evidence

An unconstrained transform used for HMC/NUTS must include the correct Jacobian in the density if the transformed variable is used as the integration variable.

## 18.3 Label switching

For additive basis mixtures, permuting component labels generally leaves the prediction unchanged.

Do not accidentally create misleading parameter summaries by averaging unlabeled components across samples.

If an ordered-center parameterization is introduced later, document how the evidence calculation and prior measure are affected.

## 18.4 Basis changes may change interpretation

A normalized Gaussian, an unnormalized Gaussian, a Lorentzian, etc. can have different meanings for the amplitude parameter.

Changing the basis should therefore trigger an explicit review of the corresponding parameter priors and constraints.

---

# 19. Definition of done

The refactor is complete when all of the following are true:

- `log_prior_theta` no longer contains distribution-specific logic;
- priors expose both `sample()` and `log_prob()`;
- basis evaluation is replaceable through a clean interface;
- likelihood is replaceable independently of basis;
- model prediction does not know about a particular prior family;
- parallel tempering does not know about a particular basis or prior;
- evidence estimation consumes generic samples/energies;
- at least two basis families can be run without modifying tempering/evidence code;
- at least two prior configurations can be swapped without modifying model/sampler code;
- regression tests show the paper-style Gaussian implementation remains numerically consistent;
- performance profiling documents the main speedups;
- HMC/NUTS can eventually plug into the same tempering interface without architectural changes.

---

# 20. Short implementation brief for the coding agent

Refactor the existing single-file implementation into a modular Bayesian spectral inference package. The most important change is to make priors first-class, composable objects with both `sample()` and `log_prob()` methods. Replace the hard-coded Gaussian prior logic in `log_prior_theta` with a generic prior interface. Separately abstract the basis function so Gaussian is just one implementation. Keep the Bayesian likelihood/posterior, evidence estimator, exchange acceptance equation, and model-selection logic generic. Make the parallel-tempering sampler accept a configurable within-temperature MCMC kernel. Preserve the current paper-faithful numerical behavior through regression tests before optimizing. After correctness is established, vectorize replica computations, cache energies/targets, reduce trace storage, and add diagnostics. Design the interfaces so a future JAX/autodiff HMC/NUTS kernel can be plugged into each temperature without changing the model, exchange, or evidence layers.
