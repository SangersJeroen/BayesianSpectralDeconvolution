# Using `bayspecdec`

A spectrum is modelled as

    y(x) = sum_k a_k * basis(x; p_k)  [+ background(x; b)]  + noise

Everything the sampler touches is a pure JAX function of one flat parameter vector `theta`, so the whole
run (leapfrog/NUTS, replica exchange, adaptation) compiles to one XLA program and gradients come from
`jax.grad`. Importing `bayspecdec` enables float64; do not override it (`sigma2 ~ 1e-4` is unreliable in float32).

## Quickstart

```python
import numpy as np
import bayspecdec as bsd

x, y, _ = bsd.make_paper_like_synthetic_data()
amplitude_prior, basis_priors = bsd.paper_synthetic_priors()

def model_for(K):
    return bsd.make_spectral_model(
        x, y, bsd.gaussian_basis, K, amplitude_prior, basis_priors, sigma2=0.01)

runs = bsd.select_model_size(
    model_for, K_values=[1, 2, 3, 4], betas=bsd.beta_schedule(24),
    burn_in=1_000, samples=1_000, config=bsd.PTConfig(kernel="nuts"))
best = min(runs, key=lambda r: r.stochastic_complexity)       # -log Z
post = best.exchange_result.samples_by_temperature[-1]        # (samples, ndim), beta = 1
```

`theta` layout: `[a_1..a_K, p1_1..p1_K, p2_1..p2_K, ..., background params, noise params]`.

## Custom basis

Any pure, differentiable `basis_fn(x, params) -> (K, n)` where `params` has shape `(n_basis_params, K)`
(broadcast over `K`; no Python loops). It may be a neural network with its weights closed over.

```python
def laplacian_basis(x, params):
    mu, b = params
    return jnp.exp(-jnp.abs(x[None, :] - mu[:, None]) / b[:, None])

bsd.make_spectral_model(x, y, laplacian_basis, K, amplitude_prior=(0, 2),
                        basis_priors=[(0, 3), bsd.GammaPrior(5.0, rate=5.0)], sigma2=0.01)
```

`basis_priors` has one prior per basis parameter, in the order `basis_fn` unpacks them.

## Priors

One prior object per parameter block, shared by all `K` components. A `(lower, upper)` tuple means uniform.
Provided: `UniformPrior`, `GammaPrior(shape, rate=… | scale=…)`, `NormalPrior(mean, std)`, `FermiDiracPrior`.
A custom prior implements the small `JaxPrior` protocol:

```python
class MyPrior:
    lower, upper = 0.0, float("inf")              # support: picks the unconstrained map
    def log_prob(self, x): ...                    # element-wise, -inf outside support, pure JAX
    def sample(self, key, shape): ...             # for chain initialisation
```

Finite box -> logistic map, `(lower, inf)` -> `lower + exp(z)`, unbounded -> identity. The sampler works in
the unconstrained `z`-space and adds `log|dtheta/dz|` itself, so chains never leave the support.

## Background

Add a non-peak component with `background_fn(x, params) -> (n,)` and one prior per parameter:

```python
model = bsd.make_spectral_model(
    x, y, bsd.gaussian_basis, K, amplitude_prior, basis_priors,
    background_fn=bsd.arctan_step_background,     # H, E0, Gamma, A, dE, omega (Kashiwamura et al., eq. 2)
    background_priors=[H, E0, Gamma, A, dE, omega],
    sigma2=...)
peaks, bg, noise = model.split_background(theta)
```

Ready-made: `arctan_step_background`, `constant_background`, `polynomial_background(degree)`. For a custom
function wrap it with `bsd.with_n_params(fn, m)` so the prior count is checked.

## Noise

Fixed: `sigma2=` (Gaussian) or `likelihood=fn(y, pred)`. Likelihoods: `gaussian_log_likelihood(sigma2)`,
`poisson_log_likelihood()`, `poisson_gaussian_log_likelihood(sigma2, half_width=12)`,
`heteroscedastic_gaussian_log_likelihood`.

Sampled: pass `noise_prior=` instead (default is `s = log sigma^2`, e.g. `log_scale_prior(1e-8, 1e-1)`);
`noise_likelihood(y, pred, s)` replaces the Gaussian. The likelihood stays fully normalised, so `log Z` is
marginalised over the noise and comparable across `K` for a fixed noise prior. `model.split(theta)` returns
`(everything but noise, noise)`.

## Without `make_spectral_model`

`make_spectral_model` only builds a `JaxModel`. The sampler uses just `ndim`, `log_likelihood(theta)`,
`log_prior(theta)` (theta-space, no Jacobian), `transform` (a `BoxTransform`, or anything with
`to_theta`/`to_z`/`log_jac`) and `sample_prior(key)`, so any layout can be written by hand.

## Kernels and adaptation

`PTConfig(kernel=...)`: `"hmc"` (fixed `num_leapfrog`), `"nuts"` (multinomial No-U-Turn, `max_tree_depth`) or
`"rwm"` (Gaussian random walk). All are vmapped over temperatures, share replica exchange (`swap_every`) and
adapt per chain: dual-averaging step size (`target_accept`) and, with `adapt_mass=True`, a Stan-style
windowed diagonal metric (needs `burn_in` of a few hundred). After `run`, `sampler.final_step_sizes` and
`sampler.final_inverse_mass` hold the tuned values in `z`-space. Exchange statistics count the sampling phase only.

## Rules for code inside the sampler

Pure functions of `theta` (no mutation or printing); no data-dependent Python `if` (use `jnp.where`);
`jnp` not `np` on traced values; static shapes (a new `K` recompiles, 1-2 s); closed-over constants such as
data or network weights are baked into the compiled program.
