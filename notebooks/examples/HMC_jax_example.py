"""
JAX-backend version of HMC_example.py: autodiff HMC + batched parallel tempering.

Run with ``python HMC_jax_example.py`` (needs ``jax``; float64 is enabled by the backend).
"""

import time

import jax.numpy as jnp
import numpy as np
import scipy as sp

import bayspecdec as bsd
from bayspecdec.jax_backend import PTConfig, make_spectral_model, select_model_size
from bayspecdec.spectrum_gen import bin_idx, lorentzian

# -- synthetic data (same as HMC_example.py) ------------------------------------------

BIN_SIZE, MIN, MAX = 1e-1, 0, 12
DET_AX = np.arange(MIN, MAX, BIN_SIZE)
PEAK = lorentzian(DET_AX, 6, 0.5, 1)

SIGNAL = np.zeros_like(DET_AX)
SIGNAL[bin_idx(DET_AX, 3)] += 1
SIGNAL[bin_idx(DET_AX, 6)] += 2
SIGNAL[bin_idx(DET_AX, 10)] += 2

BLURRED = sp.signal.convolve(SIGNAL, PEAK, mode="full")[
    len(DET_AX) // 2 : -len(DET_AX) // 2 + 1
]
y = np.random.default_rng(0).normal(loc=BLURRED, scale=0.01)


# -- model -------------------------------------------------------------------------


def blur_kernel(x, params):
    """Normalised Lorentzian, same shape as ``BlurKernel`` in HMC_example.py.

    Any differentiable JAX function with this signature works here, e.g. a
    neural network evaluated on ``x - mu``.
    """
    (mu,) = params
    return 0.5 / ((x[None, :] - mu[:, None]) ** 2 + 0.5)


def model_builder(K: int):
    return make_spectral_model(
        DET_AX,
        y,
        blur_kernel,
        K,
        amplitude_bounds=(0.2, 1.2),
        basis_param_bounds=[(1.0, 11.0)],
        sigma2=1e-4,
    )


if __name__ == "__main__":
    betas = bsd.beta_schedule(36)

    t0 = time.perf_counter()
    runs = select_model_size(
        model_builder,
        K_values=[2, 3, 4],
        betas=betas,
        burn_in=1_000,
        samples=1_000,
        config=PTConfig(num_leapfrog=10, swap_every=10),
    )
    print(f"total: {time.perf_counter() - t0:.1f}s (includes JIT compilation per K)")

    best = min(runs, key=lambda r: r.stochastic_complexity)
    print(f"best K: {best.K}, -logZ: {best.stochastic_complexity:.2f}")

    post = best.exchange_result.samples_by_temperature[-1]  # (samples, 2K) = [I..., mu...]
    mu = np.sort(post[:, best.K :], axis=1)
    print("posterior mean of sorted centres:", np.round(mu.mean(axis=0), 3))
