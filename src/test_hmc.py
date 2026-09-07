import numpy as np
from bayspecdec import (
    GaussianBasis,
    DefaultParameterization,
    paper_synthetic_prior,
    GaussianNoise,
    SpectralModel,
    HMCConfig,
    hmc_kernel_factory,
    ParallelTempering,
    beta_schedule,
    make_paper_like_synthetic_data,
)


def test_hmc():
    print("Testing HMC Integration...")
    # Generate mock data
    x, y, y_true = make_paper_like_synthetic_data(n_points=100, sigma2=0.01)

    # Setup Model
    K = 3
    model = SpectralModel(
        x=x,
        y=y,
        basis=GaussianBasis(),
        prior=paper_synthetic_prior(),
        likelihood=GaussianNoise(sigma2=0.01),
        parameterization=DefaultParameterization(K=K),
    )

    # Setup HMC Sampler
    betas = beta_schedule(2)  # small number of temps for test
    rng = np.random.default_rng(42)

    config = HMCConfig(num_steps=10, step_size=0.001)

    # We use lambda to inject the config
    sampler = ParallelTempering(
        model=model,
        betas=betas,
        rng=rng,
        kernel_factory=lambda m, b, r: hmc_kernel_factory(m, b, r, config),
    )

    print("Running Parallel Tempering with HMC...")
    result = sampler.run(burn_in=5, samples=5, swap_every=2)

    print(f"Beta=1 Acceptance Rate: {result.within_acceptance[-1]:.3f}")
    print("Success!")


if __name__ == "__main__":
    test_hmc()
