from .basis import GaussianBasis

import numpy as np

Array = np.ndarray

PAPER_AMPLITUDES = np.array([0.587, 1.522, 1.183])
PAPER_CENTERS = np.array([1.210, 1.455, 1.703])
PAPER_B = np.array([95.689, 146.837, 164.469])


def make_paper_like_synthetic_data(
    n_points: int = 301,
    sigma2: float = 0.01,
    seed: int = 7,
) -> tuple[Array, Array, Array]:
    """
    Generate the synthetic setting described in the paper.

    x = 0, 0.01, ..., 3.0  (301 points)
    sigma^2 = 0.01
    true model = 3 Gaussian bands with the paper's parameters.

    Returns x, noisy y, noiseless true y.
    """
    if n_points != 301:
        x = np.linspace(0.0, 3.0, n_points)
    else:
        x = np.arange(0.0, 3.0 + 1e-12, 0.01)

    basis = GaussianBasis()
    params = np.stack([PAPER_CENTERS, PAPER_B])
    basis_eval = basis.evaluate(x, params)
    y_true = PAPER_AMPLITUDES @ basis_eval

    rng = np.random.default_rng(seed)
    y = y_true + rng.normal(0.0, np.sqrt(sigma2), size=x.size)
    return x, y, y_true
