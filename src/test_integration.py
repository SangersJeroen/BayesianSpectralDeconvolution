import numpy as np
from bayspecdec import (
    GaussianBasis,
    LorentzianBasis,
    DefaultParameterization,
    paper_synthetic_prior,
    GaussianNoise,
    SpectralModel,
    metropolis_kernel_factory,
    ParallelTempering,
    beta_schedule,
    estimate_evidence,
    make_paper_like_synthetic_data,
)


def test_integration():
    print("Testing Integration...")
    # Generate mock data
    x, y, y_true = make_paper_like_synthetic_data(n_points=100, sigma2=0.01)

    # 1. Setup Model
    K = 3
    model = SpectralModel(
        x=x,
        y=y,
        basis=GaussianBasis(),
        prior=paper_synthetic_prior(),
        likelihood=GaussianNoise(sigma2=0.01),
        parameterization=DefaultParameterization(K=K),
    )

    # 2. Setup Sampler
    betas = beta_schedule(10)
    rng = np.random.default_rng(42)

    sampler = ParallelTempering(
        model=model, betas=betas, rng=rng, kernel_factory=metropolis_kernel_factory
    )

    # 3. Run
    print("Running Parallel Tempering...")
    result = sampler.run(burn_in=100, samples=100, swap_every=2)

    # 4. Evidence
    print("Estimating Evidence...")
    evidence = estimate_evidence(model, result)

    print(f"Log Evidence (Z): {evidence.log_z:.3f}")

    # 5. Switch basis
    model_lorentzian = SpectralModel(
        x=x,
        y=y,
        basis=LorentzianBasis(),
        prior=paper_synthetic_prior(),
        likelihood=GaussianNoise(sigma2=0.01),
        parameterization=DefaultParameterization(K=K),
    )

    sampler_l = ParallelTempering(
        model=model_lorentzian,
        betas=betas,
        rng=rng,
        kernel_factory=metropolis_kernel_factory,
    )

    print("Running Lorentzian Parallel Tempering...")
    _ = sampler_l.run(burn_in=10, samples=10, swap_every=2)
    print("Success!")


if __name__ == "__main__":
    test_integration()
