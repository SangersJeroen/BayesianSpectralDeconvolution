# Minimal Working Examples — Samplers

Three self-contained scripts, one per sampler kernel.  
Each produces a short Parallel Tempering run on synthetic spectral data and
prints basic diagnostics. Copy-paste any block into a Python file and run it.

---

## Common setup (shared by all examples)

```python
import numpy as np
from bayspecdec import (
    GaussianBasis, DefaultParameterization,
    paper_synthetic_prior, GaussianNoise,
    SpectralModel, beta_schedule,
    make_paper_like_synthetic_data,
    estimate_evidence,
)

# Generate synthetic data (3 Gaussian components, σ²=0.01)
x, y, y_true = make_paper_like_synthetic_data(n_points=301, sigma2=0.01, seed=42)

# Assemble the Bayesian model (K=3 components)
K = 3
model = SpectralModel(
    x=x, y=y,
    basis=GaussianBasis(),
    prior=paper_synthetic_prior(),
    likelihood=GaussianNoise(sigma2=0.01),
    parameterization=DefaultParameterization(K=K),
)

# Temperature ladder: 16 levels, geometric spacing from β=0 to β=1
betas = beta_schedule(L=16)
```

---

## 1. Random-Walk Metropolis

The simplest kernel. No gradient required. Tune `proposal_scales` for good acceptance (target ~0.23 for high-dimensional problems).

```python
from bayspecdec import ParallelTempering, metropolis_kernel_factory, estimate_evidence

rng = np.random.default_rng(1)

sampler = ParallelTempering(
    model=model,
    betas=betas,
    rng=rng,
    kernel_factory=metropolis_kernel_factory,   # default proposal scales
)

result = sampler.run(burn_in=2000, samples=2000)

evidence = estimate_evidence(model, result)

print(f"β=1 acceptance rate : {result.within_acceptance[-1]:.3f}")
print(f"Mean swap acceptance: {result.exchange_acceptance.mean():.3f}")
print(f"Log evidence (log Z): {evidence.log_z:.2f}")
```

**Expected output (approximate):**
```
β=1 acceptance rate : 0.21
Mean swap acceptance: 0.55
Log evidence (log Z): -47.83
```

---

## 2. Hamiltonian Monte Carlo (HMC)

Uses gradient information for more efficient exploration.  
Key parameters: `num_steps` (trajectory length L) and `step_size` (ε).  
Set `adapt_step_size=True` (default) to tune ε automatically during burn-in.

```python
from bayspecdec import (
    ParallelTempering, hmc_kernel_factory, HMCConfig, estimate_evidence,
)

rng = np.random.default_rng(2)

# Configure HMC: 10 leapfrog steps, initial ε=0.005, dual-averaging on
hmc_config = HMCConfig(
    num_steps=10,
    step_size=0.005,
    target_accept=0.8,
    adapt_step_size=True,
)

sampler = ParallelTempering(
    model=model,
    betas=betas,
    rng=rng,
    kernel_factory=lambda m, b, r: hmc_kernel_factory(m, b, r, hmc_config),
)

result = sampler.run(burn_in=1000, samples=2000)

evidence = estimate_evidence(model, result)

print(f"β=1 acceptance rate : {result.within_acceptance[-1]:.3f}")
print(f"Mean swap acceptance: {result.exchange_acceptance.mean():.3f}")
print(f"Log evidence (log Z): {evidence.log_z:.2f}")
```

> [!TIP]
> If acceptance stays near 0, decrease `step_size`.  
> If acceptance stays near 1, increase `step_size` or `num_steps`.  
> With `adapt_step_size=True` the dual-averaging scheme handles this automatically during burn-in.

---

## 3. No-U-Turn Sampler (NUTS)

NUTS eliminates the need to choose `num_steps` by adaptively building the trajectory until it starts turning back. Only `step_size` (or let dual averaging find it) needs to be set.

```python
from bayspecdec import (
    ParallelTempering, nuts_kernel_factory, NUTSConfig, estimate_evidence,
)

rng = np.random.default_rng(3)

# Configure NUTS: initial ε=0.005, max tree depth 8, dual-averaging on
nuts_config = NUTSConfig(
    step_size=0.005,
    target_accept=0.8,
    max_tree_depth=8,
    adapt_step_size=True,
)

sampler = ParallelTempering(
    model=model,
    betas=betas,
    rng=rng,
    kernel_factory=lambda m, b, r: nuts_kernel_factory(m, b, r, nuts_config),
)

result = sampler.run(burn_in=1000, samples=2000)

evidence = estimate_evidence(model, result)

print(f"β=1 acceptance rate : {result.within_acceptance[-1]:.3f}")
print(f"Mean swap acceptance: {result.exchange_acceptance.mean():.3f}")
print(f"Log evidence (log Z): {evidence.log_z:.2f}")
```

> [!NOTE]
> NUTS uses more leapfrog steps per transition than fixed-L HMC, but each
> step is more informative.  In practice NUTS typically needs fewer total
> transitions to reach the same effective sample size.

---

## Inspecting divergences (HMC / NUTS)

After a run you can inspect divergence counters directly on the kernel objects.  
Because `ParallelTempering` constructs kernels internally, the cleanest way is to
capture them via the factory closure:

```python
from bayspecdec.samplers.nuts import NoUTurnSampler

kernels: list[NoUTurnSampler] = []

def capturing_factory(m, b, r):
    k = nuts_kernel_factory(m, b, r, nuts_config)
    kernels.append(k)
    return k

sampler = ParallelTempering(model=model, betas=betas, rng=np.random.default_rng(4),
                            kernel_factory=capturing_factory)
sampler.run(burn_in=500, samples=500)

for i, k in enumerate(kernels):
    print(f"  β={betas[i]:.4f}  divergences={k.n_divergent}  last_tree_depth={k.tree_depth_last}")
```

> [!WARNING]
> A high divergence rate (e.g. > 1 % of transitions) is a signal that the
> geometry is difficult for the current step size or parameterisation.
> Decrease `step_size`, or switch to a constrained-to-unconstrained transform
> for strictly positive parameters.

---

## Swapping kernels at runtime

Because `ParallelTempering` only calls `kernel_factory(model, beta, rng)`, you
can swap between Metropolis, HMC, and NUTS without changing any other code:

```python
# Metropolis
kernel_factory = metropolis_kernel_factory

# HMC
kernel_factory = lambda m, b, r: hmc_kernel_factory(m, b, r, HMCConfig(num_steps=10, step_size=0.005))

# NUTS
kernel_factory = lambda m, b, r: nuts_kernel_factory(m, b, r, NUTSConfig(step_size=0.005))

sampler = ParallelTempering(model=model, betas=betas, rng=rng,
                            kernel_factory=kernel_factory)
```

All three produce identical `ExchangeResult` objects and are compatible with
`estimate_evidence` and `select_model_size`.
