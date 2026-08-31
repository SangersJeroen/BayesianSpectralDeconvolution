import numpy as np

Array = np.ndarray

def paper_beta_schedule(L: int) -> Array:
    """
    Temperature / inverse-temperature ladder used in Section 3.1:

        beta_1 = 0
        beta_l = 1.5 ** (l-L), l>1

    where L=24 in the synthetic experiment.
    """
    if L < 2:
        raise ValueError("Need at least two temperatures")
    beta = np.empty(L, dtype=float)
    beta[0] = 0.0
    for l in range(1, L):
        # l in the paper is 1-indexed. Here index l is 0-indexed, so paper l=l+1.
        paper_l = l + 1
        beta[l] = 1.5 ** (paper_l - L)
    beta[-1] = 1.0
    return beta
