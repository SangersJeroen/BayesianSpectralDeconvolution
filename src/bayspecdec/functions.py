import numpy as np

Array = np.ndarray

def beta_schedule(L: int) -> Array:
    """
    Temperature / inverse-temperature ladder used in Section 3.1:

        beta_1 = 0
        beta_l = 1.5 ** (l-L), l>1

    where L=24 in the synthetic experiment.
    """
    if L < 2:
        raise ValueError("Need at least two temperatures")
    beta = [0] + [1.5**(i+1-L) for i in range(1, L)]
    return np.asarray(beta)
