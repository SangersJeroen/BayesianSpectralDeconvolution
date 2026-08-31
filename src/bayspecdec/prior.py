from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticPrior:
    """Hyperparameters in Section 3.1 of the paper."""

    eta_a: float = 5.0
    lambda_a: float = 5.0
    nu0: float = 1.5
    xi0: float = 5.0
    eta_b: float = 5.0
    lambda_b: float = 0.04
