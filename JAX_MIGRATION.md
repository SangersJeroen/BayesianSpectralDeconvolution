# Migrating to the JAX backend

A prototype backend lives in `src/bayspecdec/jax_backend/` (branch `jax-backend`).
It runs next to the numpy code and is not imported by `bayspecdec`, so JAX stays optional.
This page lists what changes if you move a model, or the whole package, over to it.

## Why

Profiling the numpy HMC showed about 97% of the time in `numerical_gradient`
(about 178 model evaluations per HMC step). JAX replaces that with `jax.grad`, compiles the
leapfrog and iteration loops, and batches all temperatures with `vmap`. On the HMC example
(36 temperatures, K=2,3,4, 1000+1000 iterations) the whole selection takes about 17 s.
The numpy sampler would take several minutes for K=4 alone.
The basis can be any differentiable function, including a neural network.

## Setup changes

- Install JAX (`pip install jax`, CPU is enough for ~100-bin spectra).
- **float64 is mandatory.** `jax_backend` enables it on import (`jax_enable_x64`).
  Do not import JAX elsewhere first and then override it; `sigma2 ~ 1e-4` likelihoods are
  unreliable in float32.

## What you rewrite

| Numpy code | JAX equivalent | Notes |
|---|---|---|
| `BasisFunction.evaluate(x, params)` | `basis_fn(x, params) -> (K, n)` | Pure function, `jnp` instead of `np`, no Python loops over components (broadcast over `K`). `params` has shape `(n_basis_params, K)`. May be an NN forward pass with weights closed over. |
| `UniformPrior`, `GammaPrior` (numba `jitclass`) | `jax_backend.UniformPrior`, `jax_backend.GammaPrior` | Ported. Pass them to `make_spectral_model(amplitude_prior=..., basis_priors=[...])`; a `(lo, hi)` tuple is shorthand for uniform. `GammaPrior` takes an explicit `rate=` **or** `scale=`; see the note below. |
| Other `Prior` classes | `log_prior(theta)` and `sample_prior(key)` in a `JaxModel` | jitclass objects cannot be traced by JAX. Write a pure function, or implement the small `JaxPrior` protocol (`lower`, `upper`, `log_prob`, `sample`). |
| `Likelihood` classes | `likelihood(y, prediction) -> scalar` | `gaussian_log_likelihood(sigma2)` is provided. Poisson and Poisson+Gaussian are **not ported**. Use `jax.scipy.special.gammaln` for the factorial terms. |
| `Parameterization` (`pack`/`unpack`/`to_z`/`from_z`) | `BoxTransform` plus the `[a, p1, p2, ...]` block layout | Bounded parameters use a logistic map, positive ones `exp`. `log_jac` is added to the log density automatically. |
| `SpectralModel` | `JaxModel` / `make_spectral_model` | Same role, but holds callables instead of objects. |
| `ParallelTempering` + `hmc_kernel_factory` | `JaxParallelTempering` + `PTConfig` | See behaviour changes below. |
| `select_model_size` | `jax_backend.select_model_size` | Takes `betas` and a `PTConfig` instead of a `sampler_factory`. |
| `np.random.Generator` | `jax.random` keys | Pass `seed` to `run`; keys are split internally per chain. |

`estimate_evidence` and `ExchangeResult` are reused unchanged, so downstream analysis and plotting
keep working.

## Rules for code that runs inside the sampler

- Pure functions of `theta` only: no mutation, no side effects, no `print`.
- No data-dependent Python control flow (`if x > 0:`). Use `jnp.where` / `lax.cond`.
- No `np.` calls on traced values. Use `jnp.`.
- Shapes must be static. Changing `K` triggers a recompile (1-2 s), which is fine for model selection.
- Closed-over constants (data, NN weights) are baked into the compiled program.
  To retrain a network, rebuild the model.

## Behaviour changes to expect

- **Sampling happens in unconstrained `z`-space.** Step sizes are therefore not comparable with the
  numpy values (`step_size=0.01` there). Start from `init_step_size=0.1` and let dual averaging tune it.
- **Gamma priors use an `exp` map** (support `(0, inf)`), uniform priors a logistic map. Either way
  the sampler never leaves the support.
- **Uniform-prior bounds can no longer be crossed** during a trajectory, so there are no `-inf`
  rejections at the edge.
- **No progress bar.** The whole run is one compiled call. Compile time is included in the first run per `K`.
- **Exchange statistics count the sampling phase only.** The numpy sampler also counted burn-in.
- **Different random streams.** Results match statistically, not draw-for-draw.
- **`GammaPrior` conventions differ from numpy.** The numpy class samples and exponentiates with its
  `rate` argument as a *scale* but normalises it as a *rate*, so its `log_prob` is off by a constant
  (this shifts `log Z`). Port `GammaPrior(shape, rate=r)` from numpy as
  `GammaPrior(shape, scale=r)` to match its sampling behaviour. The JAX version is normalised correctly
  and checked against `scipy.stats.gamma`.
- **The likelihood is the corrected one.** `GaussianNoise.log_prob` in `likelihoods.py` multiplies
  terms that should be added, so numpy and JAX values of `-log Z` are not comparable until that is fixed.

## Mass-matrix adaptation

On by default (`PTConfig(adapt_mass=True)`). During warmup each temperature estimates the variance of
its own `z`-space samples in Stan-style windows (75 fast iterations, then windows of 25, 50, 100, ...,
then 50 fast iterations) and uses it as a diagonal inverse mass matrix. Step-size dual averaging restarts
after each window. Shorter warmups scale the layout to 15% / 75% / 10%, and below 20 iterations
adaptation is skipped.

- Switch it off with `adapt_mass=False`; `inverse_mass=` then sets a fixed diagonal metric for all chains.
- After `run`, the tuned values are in `sampler.final_step_sizes` `(L,)` and
  `sampler.final_inverse_mass` `(L, d)`, both in `z`-space.
- Needs `burn_in` of at least a few hundred to learn a good metric. The estimate is shrunk towards
  `1e-3` as Stan does, which sets a floor on very small variances.
- On a 3-D Gaussian with scales 0.05 / 0.5 / 5 the adapted step size at beta=1 grew from 0.003 to 0.47
  and the lag-1 autocorrelation fell from 0.90 to -0.18.

## Not yet ported

- NUTS (`samplers/nuts.py`), random-walk Metropolis and the `MCMCKernel` protocol
- Poisson and Poisson+Gaussian likelihoods; the Normal prior
- Dense mass matrices (only a diagonal metric is implemented)
- `diagnostics.py` and `spectrum_gen.py` stay numpy. They run on results, not inside the sampler.

## Suggested migration path

1. Port the basis to `jnp` and check `jax.grad` against finite differences (see `tests/test_jax_backend.py`).
2. Build the model with `make_spectral_model` (or `JaxModel` for priors other than uniform and Gamma).
3. Run `notebooks/examples/HMC_jax_example.py` as a template and compare `-log Z` and posteriors with the numpy run
   (after the likelihood fix).
4. Port the remaining likelihoods and priors as needed, then retire the numpy samplers.
