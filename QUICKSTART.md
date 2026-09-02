# Bayesian Spectral Deconvolution Quickstart

Welcome to `bayspecdec`, a modular package for Bayesian spectral deconvolution using parallel tempering (Exchange Monte Carlo). This guide will help you get a basic model up and running in minutes.

## 1. Import the Required Modules

First, let's pull in the tools we need from the package, as well as `numpy` and `matplotlib`.

```python
import numpy as np
import matplotlib.pyplot as plt

from bayspecdec import (
    make_paper_like_synthetic_data,
    GaussianBasis,
    DefaultParameterization,
    paper_synthetic_prior,
    GaussianNoise,
    SpectralModel,
    beta_schedule,
    metropolis_kernel_factory,
    ParallelTempering,
    estimate_evidence
)
```

## 2. Load or Generate Data

For this quickstart, we'll use the built-in synthetic data generator, which produces a spectrum composed of 3 overlapping Gaussian bands.

```python
x, y_noisy, y_true = make_paper_like_synthetic_data(n_points=301, sigma2=0.01)

plt.plot(x, y_noisy, '.', alpha=0.5, label="Data")
plt.plot(x, y_true, label="True Signal")
plt.legend()
plt.show()
```

## 3. Set Up the Bayesian Model

A `SpectralModel` defines the entire statistical landscape. You must specify the number of components (`K`) you want to fit. Let's fit `K=3`.

```python
K = 3

model = SpectralModel(
    x=x,
    y=y_noisy,
    basis=GaussianBasis(),
    prior=paper_synthetic_prior(),
    likelihood=GaussianNoise(sigma2=0.01),
    parameterization=DefaultParameterization(K=K)
)
```

## 4. Run Parallel Tempering (Exchange Monte Carlo)

Parallel tempering runs multiple Markov chains at different "temperatures" simultaneously. This prevents the sampler from getting trapped in local minima.

```python
# Create a temperature ladder with 16 temperatures
betas = beta_schedule(L=16)

# Initialize the parallel tempering sampler
sampler = ParallelTempering(
    model=model,
    betas=betas,
    rng=np.random.default_rng(42),
    kernel_factory=metropolis_kernel_factory
)

# Run the sampler
print("Running parallel tempering...")
result = sampler.run(
    burn_in=1000,
    samples=1000,
    swap_every=1
)
print("Done!")
```

## 5. View the Results

Now that we have our samples, we can easily evaluate the marginal likelihood (evidence). This value allows you to formally compare whether `K=3` is a better fit than `K=2` or `K=4`.

```python
evidence = estimate_evidence(model, result)
print(f"Log Evidence (Z) for K={K}: {evidence.log_z:.3f}")
```

You can also pull the "coldest" (beta=1) samples from the result object to analyze the parameter posterior distributions:

```python
# Extract parameters from the beta=1 chain
cold_samples = result.samples_by_temperature[-1]

# Unpack the first sample in the chain
a, mu, b = model.parameterization.unpack(cold_samples[0])
print(f"Sampled Centers: {mu}")
```

**Next Steps**: If you want to change the basis function, modify the priors, or run automatic model selection, see the `IN_DEPTH_GUIDE.md`!
