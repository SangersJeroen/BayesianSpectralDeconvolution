# How to Add Custom Priors

The `bayspecdec` package treats priors as first-class, composable objects. A complete model prior requires both a generative sampling method (`sample()`) for initializing chains, and a density evaluation method (`log_prob()`) used during MCMC step acceptance.

Follow these steps to define and inject custom priors into your Bayesian models.

## Step 1: Implement the `Prior` Protocol

To create a valid prior for the `SpectralModel`, your class must implement the `Prior` protocol:

```python
import numpy as np
from bayspecdec.parameters import Parameterization

class MyCustomPrior:
    def sample(self, rng: np.random.Generator, parameterization: Parameterization) -> np.ndarray:
        """Draw a sample from the prior."""
        pass

    def log_prob(self, theta: np.ndarray, parameterization: Parameterization) -> float:
        """Evaluate the log-density of the prior at theta."""
        pass
```

## Step 2: Building Composable 1D Distributions (Optional but Recommended)

It is highly recommended to build your joint parameter prior by composing 1D distribution classes. A 1D distribution should have its own `sample` and `log_prob` methods. 

For instance, if you want to add a Uniform prior constraint for the centers:

```python
import numpy as np

class UniformPrior:
    def __init__(self, lower: float, upper: float):
        self.lower = float(lower)
        self.upper = float(upper)
        
    def sample(self, rng: np.random.Generator, size=None) -> np.ndarray:
        # Draw uniform samples between lower and upper bounds
        return rng.uniform(self.lower, self.upper, size=size)
        
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        # Evaluate the log PDF. Note the vectorization!
        density = np.log(1.0 / (self.upper - self.lower))
        
        # Values outside the support bounds get log-probability of -infinity
        out_of_bounds = (x < self.lower) | (x > self.upper)
        result = np.full_like(x, density, dtype=float)
        result[out_of_bounds] = -np.inf
        
        return result
```

## Step 3: Composing the Joint Model Prior

Once you have your 1D components, you must combine them into a single joint prior that unpacks the flat `theta` vector, evaluates the individual parameter priors, and sums the log probabilities.

You can use the built-in `IndependentProductPrior` if your model assumes independence across parameter groups, or write your own.

Here is an example of creating a custom joint prior for a Gaussian spectral model, substituting our new `UniformPrior` for the center positions (`mu`), while keeping Gamma priors for amplitude (`a`) and bandwidth (`b`):

```python
from bayspecdec.priors import IndependentProductPrior, GammaPrior

def my_custom_uniform_prior() -> IndependentProductPrior:
    return IndependentProductPrior(
        amplitudes=GammaPrior(shape=5.0, rate=5.0),
        centers=UniformPrior(lower=0.0, upper=3.0),  # Our newly defined prior!
        bandwidths=GammaPrior(shape=5.0, rate=0.04)
    )
```

## Step 4: Normalization is Critical!

> [!WARNING]
> If you are doing Model Selection (estimating marginal likelihood / evidence), your priors **must be strictly normalized probability distributions**. 

Omitting normalizing constants is safe for basic MCMC parameter estimation (since they cancel out in the Metropolis-Hastings acceptance ratio), but they **do not cancel out** when computing the marginal likelihood. 

Always ensure your `log_prob` methods include the full analytic normalization constants (e.g., `-np.log(upper - lower)` for the Uniform distribution).

## Step 5: Injecting the Prior into the Model

Simply pass the constructed prior instance when instantiating the `SpectralModel`:

```python
from bayspecdec import SpectralModel, GaussianBasis, DefaultParameterization, GaussianNoise

model = SpectralModel(
    x=x_data,
    y=y_data,
    basis=GaussianBasis(),
    prior=my_custom_uniform_prior(),  # Inject custom prior here
    likelihood=GaussianNoise(sigma2=0.01),
    parameterization=DefaultParameterization(K=3)
)
```

The sampler will automatically query your prior for proper initialization and likelihood weighting during the parallel tempering run!
