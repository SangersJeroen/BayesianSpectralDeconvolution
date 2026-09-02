# In-Depth Guide to `bayspecdec`

The `bayspecdec` package is built around a highly modular, decoupled architecture. This allows advanced users to rapidly iterate on scientific hypotheses (like swapping the underlying spectral basis or applying constrained priors) without having to touch the complex underlying MCMC algorithms or evidence estimation math.

This document serves as an exhaustive reference for what is possible within the framework.

---

## 1. Architectural Overview

At the heart of the package is the `SpectralModel`. The model doesn't "know" how to evaluate a Gaussian, nor does it "know" what a Gamma prior is. Instead, it acts as a coordinator between four distinct objects:

1. **BasisFunction**: Defines the functional form of a single spectral component.
2. **Prior**: Defines the joint probability density over the entire parameter space.
3. **Likelihood**: Defines the observation model (e.g., how the true spectrum is corrupted by noise).
4. **Parameterization**: Defines how the structured parameters (like amplitudes and centers) are packed into a flat 1D array (`theta`) for the sampler, and how they are unpacked back into structured forms.

### Why this design?
In spectral deconvolution, the mathematical form of the likelihood and the evidence integral are agnostic to the basis. By enforcing these boundaries, you can cleanly calculate the marginal likelihood (evidence) for a Lorentzian model, compare it to a Gaussian model, and know with absolute certainty that any difference in evidence is due to the basis physics, not a discrepancy in how the sampler was hardcoded.

---

## 2. Advanced Model Construction

### Swapping the Basis Function
By default, the package provides `GaussianBasis` and `LorentzianBasis`. 

```python
from bayspecdec import SpectralModel, LorentzianBasis

# Simply swap the basis when instantiating the model
model = SpectralModel(
    ...,
    basis=LorentzianBasis(),
    ...
)
```
For information on building your own, see [CUSTOM_BASIS.md](CUSTOM_BASIS.md).

### Designing Custom Priors
The paper uses independent Gamma priors for amplitudes, Normal priors for centers, and Gamma priors for bandwidths. You can override any of these.

```python
from bayspecdec.priors import IndependentProductPrior, GammaPrior, NormalPrior

# Define a tighter prior on the centers
custom_prior = IndependentProductPrior(
    amplitudes=GammaPrior(shape=5.0, rate=5.0),
    centers=NormalPrior(mean=1.5, precision=20.0), # Tighter precision
    bandwidths=GammaPrior(shape=5.0, rate=0.04)
)
```
For deep-dives on adding entirely new distribution families and ensuring proper normalizations, see [CUSTOM_PRIOR.md](CUSTOM_PRIOR.md).

---

## 3. Parallel Tempering (Exchange Monte Carlo)

Because spectral posteriors are notoriously multi-modal (due to component label-switching and isolated local minima), the package uses Parallel Tempering.

### The Temperature Ladder
The `beta_schedule(L)` function generates a geometric temperature ladder. `beta=1.0` is the "cold" true posterior, and `beta=0.0` is the "hot" prior.

### The Sampler Kernel
Currently, the package ships with a `RandomWalkMetropolis` kernel. The tempering layer is entirely agnostic to this!

```python
from bayspecdec import ParallelTempering, metropolis_kernel_factory

sampler = ParallelTempering(
    model=model,
    betas=betas,
    rng=np.random.default_rng(),
    kernel_factory=metropolis_kernel_factory
)
```
*Note: Because of this decoupled design, future implementations of Hamiltonian Monte Carlo (HMC) or the No-U-Turn Sampler (NUTS) can be injected cleanly by simply writing a new `nuts_kernel_factory`.*

---

## 4. Automatic Model Selection

The most powerful feature of the Bayesian approach is objectively selecting the number of components (`K`). The package provides a wrapper that runs Parallel Tempering over multiple `K` values and calculates the stochastic complexity (`-log Z`) for each.

```python
from bayspecdec import select_model_size, DefaultParameterization, GaussianBasis, paper_synthetic_prior, GaussianNoise

def my_model_factory(K: int):
    return SpectralModel(
        x=x, y=y,
        basis=GaussianBasis(),
        prior=paper_synthetic_prior(),
        likelihood=GaussianNoise(sigma2=0.01),
        parameterization=DefaultParameterization(K=K)
    )

def my_sampler_factory(model, rng):
    return ParallelTempering(
        model=model, betas=beta_schedule(16), rng=rng, kernel_factory=metropolis_kernel_factory
    )

# Evaluate K=1 through K=5
runs = select_model_size(
    model_factory=my_model_factory,
    K_values=[1, 2, 3, 4, 5],
    sampler_factory=my_sampler_factory,
    burn_in=2000,
    samples=2000
)

# You can access the evidence for K=3:
run_K3 = [r for r in runs if r.K == 3][0]
print(run_K3.evidence.log_z)
```

The model with the **lowest stochastic complexity** (or highest `log Z`) is mathematically the most probable model given the data, naturally penalizing overfitting.

---

## 5. Diagnostics and Visualization

MCMC algorithms require careful monitoring to ensure convergence. The `bayspecdec.diagnostics` module provides tools to assess chain health.

### Exchange Acceptance
If adjacent temperature chains do not swap frequently, the cold chain won't benefit from the exploration of the hot chains. 

```python
from bayspecdec.diagnostics import plot_exchange_acceptance
plot_exchange_acceptance(result)
```
If you see acceptance rates near 0% for certain temperature gaps, you need to increase `L` (the number of temperatures) to close the thermodynamic distance between replicas.

### Energy Traces
Energy traces (the negative log-likelihood) should show the hot chains wildly exploring parameter space (high variance) while the cold chains settle into deep energy wells.

```python
from bayspecdec.diagnostics import plot_energy_traces
plot_energy_traces(result, beta_indices=[0, 8, 15]) # Plot hot, warm, and cold
```

---

## 6. Extracting the Posterior Mode (MAP Estimate)

If you need a single "best fit" spectrum to plot alongside your data, you can extract the Maximum A Posteriori (MAP) estimate from your samples:

```python
# The ModelRun object from select_model_size automatically caches this
best_theta = run_K3.posterior_mode_sample

# Reconstruct the predictive spectrum
y_pred = run_K3.model.predict(best_theta)

import matplotlib.pyplot as plt
plt.plot(x, y, '.', alpha=0.5)
plt.plot(x, y_pred, 'k-', linewidth=2, label=f"MAP Fit (K={run_K3.K})")
plt.legend()
plt.show()
```
