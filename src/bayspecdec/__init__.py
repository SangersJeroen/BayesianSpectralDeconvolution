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
"""
