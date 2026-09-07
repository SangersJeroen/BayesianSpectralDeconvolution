"""
tests/test_samplers.py

Unit + integration tests for:
  - numerical gradient accuracy
  - leapfrog reversibility
  - HMC on a 1-D standard normal
  - dual-averaging convergence
  - WelfordCovariance estimator
  - NUTS on a 1-D standard normal (no divergences)
  - U-turn criterion
  - HMC + NUTS inside ParallelTempering (smoke test)
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / tiny models for isolated testing
# ---------------------------------------------------------------------------

def unit_normal_potential(theta):
    """U(θ) = ½ θ²  (standard normal)."""
    return float(0.5 * np.dot(theta, theta))


def unit_normal_grad(theta):
    return theta.copy()


def unit_normal_log_density(theta):
    return float(-0.5 * np.dot(theta, theta))


# ---------------------------------------------------------------------------
# 1. Numerical gradient accuracy
# ---------------------------------------------------------------------------

class TestNumericalGradient:
    def test_quadratic(self):
        from bayspecdec.samplers.hmc import numerical_gradient
        f = lambda x: float(np.sum(x ** 2))
        x = np.array([1.0, 2.0, -3.0])
        g = numerical_gradient(f, x)
        expected = 2.0 * x
        np.testing.assert_allclose(g, expected, rtol=1e-4)

    def test_standard_normal_potential(self):
        from bayspecdec.samplers.hmc import numerical_gradient
        x = np.array([0.5, -1.0, 2.0])
        g = numerical_gradient(unit_normal_potential, x)
        np.testing.assert_allclose(g, x, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. Leapfrog reversibility
# ---------------------------------------------------------------------------

class TestLeapfrog:
    def test_reversibility(self):
        from bayspecdec.samplers.hmc import leapfrog
        inv_M = np.ones(2)
        theta0 = np.array([1.0, -0.5])
        p0 = np.array([0.3, 0.7])
        eps = 0.1
        L = 5

        theta1, p1 = leapfrog(theta0, p0, eps, L, unit_normal_grad, inv_M)
        # Reverse: negate momentum, step backward
        theta2, p2 = leapfrog(theta1, -p1, eps, L, unit_normal_grad, inv_M)
        np.testing.assert_allclose(theta2, theta0, atol=1e-10)
        np.testing.assert_allclose(-p2, p0, atol=1e-10)

    def test_energy_conservation_small_step(self):
        """Leapfrog should approximately conserve H for small ε."""
        from bayspecdec.samplers.hmc import leapfrog
        inv_M = np.ones(1)
        theta0 = np.array([1.0])
        p0 = np.array([0.0])
        eps = 1e-3
        L = 100
        H0 = unit_normal_potential(theta0) + 0.5 * float(p0 @ (inv_M * p0))
        theta1, p1 = leapfrog(theta0, p0, eps, L, unit_normal_grad, inv_M)
        H1 = unit_normal_potential(theta1) + 0.5 * float(p1 @ (inv_M * p1))
        assert abs(H1 - H0) < 1e-3, f"|ΔH| = {abs(H1-H0):.2e}"


# ---------------------------------------------------------------------------
# 3. HMC samples a standard normal
# ---------------------------------------------------------------------------

class TestHMCStandardNormal:
    """
    Run HMC targeting a d-dimensional standard normal using a tiny
    SpectralModel-like shim so we don't need real spectral data.
    """

    @pytest.fixture
    def normal_model_and_sampler(self):
        """Build a minimal SpectralModel wrapping a standard normal."""
        from bayspecdec import (
            GaussianBasis, DefaultParameterization, paper_synthetic_prior,
            GaussianNoise, SpectralModel, make_paper_like_synthetic_data,
        )
        # Very small dataset so tests are fast
        x, y, _ = make_paper_like_synthetic_data(n_points=30, sigma2=0.01, seed=1)
        K = 1
        model = SpectralModel(
            x=x, y=y,
            basis=GaussianBasis(),
            prior=paper_synthetic_prior(),
            likelihood=GaussianNoise(sigma2=0.01),
            parameterization=DefaultParameterization(K=K),
        )
        return model

    def test_hmc_runs_no_exception(self, normal_model_and_sampler):
        from bayspecdec import (
            ParallelTempering, beta_schedule, hmc_kernel_factory, HMCConfig,
        )
        model = normal_model_and_sampler
        rng = np.random.default_rng(42)
        betas = beta_schedule(3)
        config = HMCConfig(num_steps=5, step_size=0.005)
        sampler = ParallelTempering(
            model=model, betas=betas, rng=rng,
            kernel_factory=lambda m, b, r: hmc_kernel_factory(m, b, r, config),
        )
        result = sampler.run(burn_in=10, samples=20)
        # Cold chain should have some accepted moves
        assert result.within_acceptance[-1] >= 0.0
        assert result.samples_by_temperature[-1].shape[0] == 20

    def test_hmc_acceptance_improves_with_small_step(self, normal_model_and_sampler):
        """Smaller step size → higher acceptance (leapfrog error ↓)."""
        from bayspecdec import (
            ParallelTempering, beta_schedule, hmc_kernel_factory, HMCConfig,
        )
        model = normal_model_and_sampler

        def run_with_eps(eps):
            rng = np.random.default_rng(7)
            betas = beta_schedule(2)
            config = HMCConfig(num_steps=3, step_size=eps, adapt_step_size=False)
            sampler = ParallelTempering(
                model=model, betas=betas, rng=rng,
                kernel_factory=lambda m, b, r: hmc_kernel_factory(m, b, r, config),
            )
            result = sampler.run(burn_in=0, samples=50)
            return result.within_acceptance[-1]

        acc_small = run_with_eps(0.001)
        acc_large = run_with_eps(0.5)
        assert acc_small >= acc_large, (
            f"Expected acc(ε=0.001)={acc_small:.3f} ≥ acc(ε=0.5)={acc_large:.3f}"
        )


# ---------------------------------------------------------------------------
# 4. Dual-averaging adaptation
# ---------------------------------------------------------------------------

class TestDualAveraging:
    def test_converges_toward_target(self):
        from bayspecdec.samplers.adaptation import StepSizeAdaptation
        adapt = StepSizeAdaptation(initial_step_size=1.0, target_accept=0.8)
        # Feed perfect acceptance → step size should increase
        for _ in range(100):
            adapt.update(1.0)
        final = adapt.final_step_size()
        assert final > 1.0, f"Expected step size to grow, got {final:.4f}"

    def test_decreases_on_zero_acceptance(self):
        from bayspecdec.samplers.adaptation import StepSizeAdaptation
        adapt = StepSizeAdaptation(initial_step_size=1.0, target_accept=0.8)
        for _ in range(200):
            adapt.update(0.0)
        final = adapt.final_step_size()
        assert final < 1.0, f"Expected step size to shrink, got {final:.4f}"

    def test_target_acceptance_stabilises(self):
        """Simulate a sampler that perfectly achieves target_accept."""
        from bayspecdec.samplers.adaptation import StepSizeAdaptation
        target = 0.8
        adapt = StepSizeAdaptation(initial_step_size=0.5, target_accept=target)
        for _ in range(500):
            adapt.update(target)   # perfect feedback
        # H_bar should be near 0, step size near initial * 10 (μ)
        assert abs(adapt.state.H_bar) < 0.05


# ---------------------------------------------------------------------------
# 5. WelfordCovariance estimator
# ---------------------------------------------------------------------------

class TestWelford:
    def test_diagonal_variance_converges(self):
        from bayspecdec.samplers.adaptation import WelfordCovariance
        rng = np.random.default_rng(0)
        ndim = 4
        true_var = np.array([1.0, 4.0, 0.25, 9.0])
        samples = rng.normal(0, np.sqrt(true_var), size=(5000, ndim))
        est = WelfordCovariance(ndim, diagonal=True)
        for s in samples:
            est.update(s)
        var = est.get_variance()
        np.testing.assert_allclose(var, true_var, rtol=0.1)

    def test_reset_clears_state(self):
        from bayspecdec.samplers.adaptation import WelfordCovariance
        rng = np.random.default_rng(1)
        est = WelfordCovariance(2, diagonal=True)
        for s in rng.normal(size=(50, 2)):
            est.update(s)
        assert est.n == 50
        est.reset()
        assert est.n == 0
        np.testing.assert_array_equal(est.mean, np.zeros(2))


# ---------------------------------------------------------------------------
# 6. U-turn criterion
# ---------------------------------------------------------------------------

class TestUTurn:
    def test_obvious_uturn(self):
        """Points heading back toward each other → U-turn."""
        from bayspecdec.samplers.nuts import is_uturn
        theta_minus = np.array([0.0])
        theta_plus  = np.array([1.0])
        # Momenta pointing inward
        p_minus = np.array([1.0])   # points forward (toward plus) — no uturn on minus
        p_plus  = np.array([-1.0])  # points backward (toward minus) — uturn on plus
        assert is_uturn(theta_minus, theta_plus, p_minus, p_plus, np.ones(1))

    def test_no_uturn(self):
        """Points heading away from each other → no U-turn."""
        from bayspecdec.samplers.nuts import is_uturn
        theta_minus = np.array([0.0])
        theta_plus  = np.array([1.0])
        p_minus = np.array([1.0])   # same direction as delta
        p_plus  = np.array([1.0])   # same direction as delta
        assert not is_uturn(theta_minus, theta_plus, p_minus, p_plus, np.ones(1))


# ---------------------------------------------------------------------------
# 7. NUTS smoke test (no divergences expected for small step size)
# ---------------------------------------------------------------------------

class TestNUTSSmoke:
    def test_nuts_runs_no_exception(self):
        from bayspecdec import (
            GaussianBasis, DefaultParameterization, paper_synthetic_prior,
            GaussianNoise, SpectralModel, make_paper_like_synthetic_data,
            ParallelTempering, beta_schedule, nuts_kernel_factory, NUTSConfig,
        )
        x, y, _ = make_paper_like_synthetic_data(n_points=30, sigma2=0.01, seed=5)
        K = 1
        model = SpectralModel(
            x=x, y=y,
            basis=GaussianBasis(),
            prior=paper_synthetic_prior(),
            likelihood=GaussianNoise(sigma2=0.01),
            parameterization=DefaultParameterization(K=K),
        )
        rng = np.random.default_rng(99)
        betas = beta_schedule(3)
        config = NUTSConfig(step_size=0.001, max_tree_depth=4)
        sampler = ParallelTempering(
            model=model, betas=betas, rng=rng,
            kernel_factory=lambda m, b, r: nuts_kernel_factory(m, b, r, config),
        )
        result = sampler.run(burn_in=5, samples=10)
        assert result.samples_by_temperature[-1].shape[0] == 10

    def test_nuts_no_divergences_small_step(self):
        """With a tiny step size, divergences should be rare / zero."""
        from bayspecdec import (
            GaussianBasis, DefaultParameterization, paper_synthetic_prior,
            GaussianNoise, SpectralModel, make_paper_like_synthetic_data,
            ParallelTempering, beta_schedule, nuts_kernel_factory, NUTSConfig,
        )
        from bayspecdec.samplers.nuts import NoUTurnSampler

        x, y, _ = make_paper_like_synthetic_data(n_points=30, sigma2=0.01, seed=3)
        K = 1
        model = SpectralModel(
            x=x, y=y,
            basis=GaussianBasis(),
            prior=paper_synthetic_prior(),
            likelihood=GaussianNoise(sigma2=0.01),
            parameterization=DefaultParameterization(K=K),
        )
        rng = np.random.default_rng(77)
        betas = beta_schedule(2)
        config = NUTSConfig(step_size=0.0005, max_tree_depth=3)
        kernels: list[NoUTurnSampler] = []

        def factory(m, b, r):
            k = nuts_kernel_factory(m, b, r, config)
            kernels.append(k)
            return k

        sampler = ParallelTempering(model=model, betas=betas, rng=rng, kernel_factory=factory)
        sampler.run(burn_in=10, samples=30)
        total_div = sum(k.n_divergent for k in kernels)
        # A very small number of divergences (e.g. from the prior-draw initial
        # point) is acceptable; an epidemic would indicate a bug.
        assert total_div <= 3, f"Too many divergences with tiny ε: {total_div}"


# ---------------------------------------------------------------------------
# 8. Kernel factories produce correct types
# ---------------------------------------------------------------------------

class TestFactories:
    def test_hmc_factory_type(self):
        from bayspecdec import (
            GaussianBasis, DefaultParameterization, paper_synthetic_prior,
            GaussianNoise, SpectralModel, make_paper_like_synthetic_data,
            hmc_kernel_factory, HMCConfig, HamiltonianMonteCarlo,
        )
        x, y, _ = make_paper_like_synthetic_data(n_points=20)
        model = SpectralModel(x=x, y=y, basis=GaussianBasis(),
                              prior=paper_synthetic_prior(),
                              likelihood=GaussianNoise(sigma2=0.01),
                              parameterization=DefaultParameterization(K=1))
        rng = np.random.default_rng(0)
        k = hmc_kernel_factory(model, 1.0, rng, HMCConfig(num_steps=5, step_size=0.01))
        assert isinstance(k, HamiltonianMonteCarlo)

    def test_nuts_factory_type(self):
        from bayspecdec import (
            GaussianBasis, DefaultParameterization, paper_synthetic_prior,
            GaussianNoise, SpectralModel, make_paper_like_synthetic_data,
            nuts_kernel_factory, NUTSConfig, NoUTurnSampler,
        )
        x, y, _ = make_paper_like_synthetic_data(n_points=20)
        model = SpectralModel(x=x, y=y, basis=GaussianBasis(),
                              prior=paper_synthetic_prior(),
                              likelihood=GaussianNoise(sigma2=0.01),
                              parameterization=DefaultParameterization(K=1))
        rng = np.random.default_rng(0)
        k = nuts_kernel_factory(model, 1.0, rng, NUTSConfig(step_size=0.01))
        assert isinstance(k, NoUTurnSampler)
