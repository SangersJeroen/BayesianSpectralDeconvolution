# How to Add a Custom Basis

The `bayspecdec` package is designed so that you can add new spectral basis functions (e.g., Voigt, pseudo-Voigt, skewed Gaussian) without modifying the samplers, evidence estimators, or parallel tempering logic.

Follow these steps to add a new custom basis to the framework.

## Step 1: Implement the `BasisFunction` Protocol

A basis function is defined as a Python class that satisfies the `BasisFunction` protocol. Create a new class in `bayspecdec/basis.py` (or your own separate file).

Your class needs to implement three things:
1. `name`: A string identifier.
2. `n_parameters_per_basis`: An integer specifying how many parameters **each individual component** requires, *excluding* the amplitude (which is handled automatically by the model).
3. `evaluate(self, x: np.ndarray, params: np.ndarray) -> np.ndarray`: A method that vectorizes evaluation over `K` components and `n` data points.

### Example: A Laplacian Basis

Suppose we want to add a Laplacian (Double Exponential) basis function:

$$ \phi_k(x) = \exp\left( - \frac{|x - \mu_k|}{b_k} \right) $$

```python
import numpy as np

class LaplacianBasis:
    name = "Laplacian"
    
    # Each component requires 2 parameters: a center (mu) and a scale (b)
    n_parameters_per_basis = 2 

    def evaluate(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate K Laplacian basis components at n points.
        
        Args:
            x: Array of shape (n,) containing the observation grid.
            params: Array of shape (2, K) containing (mu, b) for each of the K components.
            
        Returns:
            An array of shape (K, n) containing the evaluated basis functions.
        """
        # Unpack the parameters. The order must match your Parameterization!
        mu = params[0]  # shape (K,)
        b = params[1]   # shape (K,)
        
        # Use NumPy broadcasting to evaluate all K components over all n points simultaneously.
        # x[None, :] has shape (1, n)
        # mu[:, None] has shape (K, 1)
        # Resulting shape will be (K, n)
        return np.exp(- np.abs(x[None, :] - mu[:, None]) / b[:, None])
```

## Step 2: Ensure Parameterization Matches

The parameters supplied to `evaluate(x, params)` come directly from the `Parameterization` you supply to the `SpectralModel`. 

If you use the `DefaultParameterization`, it unpacks the full state vector `theta` into `(a, mu, b)`. 
- The model extracts `a` as the amplitudes. 
- It passes the remaining tuple `(mu, b)` to your basis function by stacking them into an array of shape `(2, K)`. 

If your new basis requires a different number of parameters (e.g., a Voigt profile requiring `mu`, `sigma`, and `gamma`), you must define a custom `Parameterization` class that unpacks `theta` into `(a, mu, sigma, gamma)`.

## Step 3: Inject the Basis into the Model

You can now use your custom basis immediately by passing it to the `SpectralModel`:

```python
from bayspecdec import SpectralModel

# Assuming LaplacianBasis is defined as above
model = SpectralModel(
    x=x_data,
    y=y_data,
    basis=LaplacianBasis(),
    prior=my_prior,
    likelihood=my_likelihood,
    parameterization=my_parameterization
)
```

The sampler will automatically query the basis using the new `evaluate` method and perform inference!
