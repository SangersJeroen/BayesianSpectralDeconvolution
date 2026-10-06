"""Numpy backend: transforms, priors, and the sampled-noise (sigma^2) hyperparameter."""

import numpy as np
import pytest
from scipy import stats

from bayspecdec import (
    DefaultParameterization,
    GaussianBasis,
    GaussianNoise,
    HMCConfig,
    NUTSConfig,
    ParallelTempering,
    SpectralModel,
    hmc_kernel_factory,
    metropolis_kernel_factory,
    nuts_kernel_factory,
)
from bayspecdec.parameters import BlockParameterization
from bayspecdec.priors import (
    FermiDiracPrior,
    GammaPrior,
    IndependentProductPrior,
    NormalPrior,
    ProductPrior,
    UniformPrior,
    log_scale_prior,
)
from bayspecdec.samplers.metropolis import MetropolisState

# -- parameterization -------------------------------------------------------------------------


def test_block_parameterization_roundtrip_layout_and_jacobian():
    p = BlockParameterization(
        2, [(0.0, np.inf), (1.0, 3.0), (-np.inf, np.inf)], noise_bounds=[(-5.0, -1.0)]
    )
    assert p.ndim == 3 * 2 + 1 and p.n_noise == 1
    z = np.array([-0.3, 0.8, 0.4, -1.2, 0.5, -0.7, 0.9])
    theta = p.from_z(z)
    np.testing.assert_allclose(p.to_z(theta), z, atol=1e-9)
    a, m, c = p.unpack(theta)
    assert a.shape == m.shape == c.shape == (2,)
    assert np.all(a > 0) and np.all((m > 1) & (m < 3)) and -5 < p.noise(theta)[0] < -1

    # log_jacobian == log |det d theta / d z| (diagonal map)
    h = 1e-6
    diag = [(p.from_z(z + h * e)[i] - p.from_z(z - h * e)[i]) / (2 * h) for i, e in enumerate(np.eye(7))]
    np.testing.assert_allclose(p.log_jacobian(theta), np.sum(np.log(diag)), atol=1e-6)
    with pytest.raises(ValueError):
        p.validate(theta[:-1])


def test_default_parameterization_matches_old_layout_without_noise():
    p = DefaultParameterization(K=3)
    theta = np.arange(1.0, 10.0)
    a, mu, b = p.unpack(theta)
    np.testing.assert_array_equal(a, [1, 2, 3])
    np.testing.assert_array_equal(mu, [4, 5, 6])
    np.testing.assert_array_equal(b, [7, 8, 9])
    assert p.ndim == 9 and p.noise(theta).size == 0
    q = DefaultParameterization(K=3, n_noise=1, noise_bounds=[(-4.0, 0.0)])
    assert q.ndim == 10 and q.unpack(np.arange(10.0))[2].size == 3  # noise not leaked into b


# -- priors ---------------------------------------------------------------------------------------


def test_numpy_priors_are_normalised_densities():
    from scipy.integrate import quad

    x = np.linspace(-6, 9, 31)
    np.testing.assert_allclose(NormalPrior(1.5, 2.0).log_prob(x), stats.norm.logpdf(x, 1.5, 2.0))
    xp = np.linspace(0.1, 30, 20)
    np.testing.assert_allclose(GammaPrior(3.0, 2.0).log_prob(xp), stats.gamma.logpdf(xp, 3.0, scale=0.5))
    u = UniformPrior(1.0, 3.0)
    np.testing.assert_allclose(u.log_prob(np.array([0.5, 1.0, 2.0, 3.0, 3.5])), [-np.inf, np.log(0.5), np.log(0.5), np.log(0.5), -np.inf])
    f = FermiDiracPrior(1.0, 0.2)
    total = quad(lambda t: float(np.exp(f.log_prob(np.array([t]))[0])), 0.0, 40.0, points=[1.0])[0]
    np.testing.assert_allclose(total, 1.0, rtol=1e-6)


def test_product_prior_with_noise_block():
    noise = log_scale_prior(1e-6, 1.0)
    prior = IndependentProductPrior(GammaPrior(5.0, 5.0), NormalPrior(1.5, 5.0), GammaPrior(5.0, 0.04), noise=noise)
    param = DefaultParameterization(2, n_noise=1, noise_bounds=prior.noise_bounds())
    theta = prior.sample(np.random.default_rng(0), param)
    assert theta.size == param.ndim == 7
    assert np.log(1e-6) <= theta[-1] <= 0.0
    lp = prior.log_prob(theta, param)
    ref = (
        GammaPrior(5.0, 5.0).log_prob(theta[:2]).sum()
        + NormalPrior(1.5, 5.0).log_prob(theta[2:4]).sum()
        + GammaPrior(5.0, 0.04).log_prob(theta[4:6]).sum()
        + noise.log_prob(theta[6:7]).sum()
    )
    np.testing.assert_allclose(lp, ref)
    assert prior.log_prob(np.r_[theta[:6], 5.0], param) == -np.inf  # sigma^2 above the prior box


# -- likelihood -----------------------------------------------------------------------------------


def test_gaussian_noise_matches_scipy_fixed_and_sampled():
    rng = np.random.default_rng(0)
    y, pred = rng.normal(size=40), rng.normal(size=40)
    ref = stats.norm.logpdf(y, pred, np.sqrt(0.3)).sum()
    np.testing.assert_allclose(GaussianNoise(0.3).log_prob(y, pred), ref)
    sampled = GaussianNoise(None)
    np.testing.assert_allclose(sampled.log_prob(y, pred, {"noise": np.array([np.log(0.3)])}), ref)
    np.testing.assert_allclose(sampled.energy(y, pred, {"noise": np.array([np.log(0.3)])}), -ref)
    with pytest.raises(ValueError):
        sampled.log_prob(y, pred)


def _model(sigma2_true=0.01, n=60, sampled=True, seed=0, K=1):
    x = np.linspace(0.0, 3.0, n)
    rng = np.random.default_rng(seed)
    y = 1.0 * np.exp(-0.5 * 60.0 * (x - 1.5) ** 2) + rng.normal(0, np.sqrt(sigma2_true), n)
    noise = log_scale_prior(1e-5, 1.0)
    prior = IndependentProductPrior(
        GammaPrior(5.0, 5.0), UniformPrior(0.5, 2.5), GammaPrior(5.0, 0.05),
        noise=noise if sampled else None,
    )
    param = DefaultParameterization(
        K, n_noise=1 if sampled else 0,
        noise_bounds=prior.noise_bounds() if sampled else None,
        mu_bounds=(0.5, 2.5),
    )
    lik = GaussianNoise(None if sampled else sigma2_true)
    return SpectralModel(x, y, GaussianBasis(), prior, lik, param)


def test_spectral_model_noise_plumbing_and_validation():
    m = _model()
    theta = m.prior.sample(np.random.default_rng(1), m.parameterization)
    theta_lo, theta_hi = theta.copy(), theta.copy()
    theta_lo[-1], theta_hi[-1] = np.log(1e-3), np.log(1e-1)
    assert m.log_likelihood(theta_lo) != m.log_likelihood(theta_hi)
    assert m.predict(theta).shape == (60,)
    z = m.parameterization.to_z(theta)
    assert np.isfinite(m.log_target_z(z, 1.0))

    # a fixed-sigma model with the same parameters must equal the sampled one at that sigma
    fixed = SpectralModel(
        m.x, m.y, GaussianBasis(), m.prior, GaussianNoise(1e-3),
        DefaultParameterization(1, mu_bounds=(0.5, 2.5)),
    )
    th_fixed = theta_lo[:-1]
    np.testing.assert_allclose(m.log_likelihood(theta_lo), fixed.log_likelihood(th_fixed))

    with pytest.raises(ValueError, match="noise hyperparameter"):
        SpectralModel(m.x, m.y, GaussianBasis(), m.prior, GaussianNoise(0.01), m.parameterization)
    with pytest.raises(ValueError, match="noise hyperparameter"):
        SpectralModel(m.x, m.y, GaussianBasis(), m.prior, GaussianNoise(None), DefaultParameterization(1))


# -- samplers on a Jacobian-sensitive target and on the noise model --------------------------------


class _UniformBoxModel:
    """Flat prior on (0, 1) x (-2, 2): in theta-space the samples must be uniform."""

    def __init__(self):
        self.parameterization = BlockParameterization(1, [(0.0, 1.0), (-2.0, 2.0)])

    def log_tempered_target(self, theta, beta):
        inside = 0 < theta[0] < 1 and -2 < theta[1] < 2
        return 0.0 if inside else -np.inf

    def log_target_z(self, z, beta):
        theta = self.parameterization.from_z(z)
        return self.log_tempered_target(theta, beta) + self.parameterization.log_jacobian(theta)

    def energy(self, theta):
        return 0.0


@pytest.mark.parametrize("kind", ["hmc", "nuts", "rwm"])
def test_numpy_kernels_sample_uniform_box_uniformly(kind):
    model = _UniformBoxModel()
    rng = np.random.default_rng(3)
    if kind == "hmc":
        kernel = hmc_kernel_factory(model, 1.0, rng, HMCConfig(num_steps=5, step_size=0.3, adapt_step_size=False))
    elif kind == "nuts":
        kernel = nuts_kernel_factory(model, 1.0, rng, NUTSConfig(step_size=0.4, max_tree_depth=4, adapt_step_size=False))
    else:
        kernel = metropolis_kernel_factory(model, 1.0, rng, proposal_scales=np.array([1.0, 1.0]))
    state = MetropolisState(theta=np.array([0.5, 0.0]), log_target=0.0, energy=0.0)
    draws = []
    for i in range(3000):
        kernel.step(state)
        draws.append(state.theta.copy())
    d = np.array(draws[200:])
    assert stats.kstest(d[:, 0], "uniform", args=(0, 1)).pvalue > 1e-3 or abs(d[:, 0].mean() - 0.5) < 0.04
    assert abs(d[:, 0].mean() - 0.5) < 0.05 and abs(d[:, 0].var() - 1 / 12) < 0.02
    assert abs(d[:, 1].mean()) < 0.2 and abs(d[:, 1].var() - 16 / 12) < 0.35
    assert state.accepted > 500


@pytest.mark.parametrize(
    "factory",
    [
        lambda m, b, r: hmc_kernel_factory(m, b, r, HMCConfig(num_steps=8, step_size=0.05)),
        lambda m, b, r: nuts_kernel_factory(m, b, r, NUTSConfig(step_size=0.05, max_tree_depth=5)),
        lambda m, b, r: metropolis_kernel_factory(m, b, r, proposal_scales=np.full(4, 0.04)),
    ],
    ids=["hmc", "nuts", "rwm"],
)
def test_numpy_sampler_recovers_sigma2(factory):
    true_s2 = 0.01
    model = _model(true_s2, n=80)
    sampler = ParallelTempering(
        model, 1.5 ** (np.arange(-6, 1)), np.random.default_rng(5), kernel_factory=factory
    )
    result = sampler.run(burn_in=250, samples=400, swap_every=2)
    post = result.samples_by_temperature[-1]
    assert post.shape == (400, 4)
    s2 = np.exp(post[:, -1])
    assert 0.5 * true_s2 < s2.mean() < 2.0 * true_s2
    assert abs(post[:, 1].mean() - 1.5) < 0.1  # centre recovered
    assert result.within_acceptance[-1] > 0.05
