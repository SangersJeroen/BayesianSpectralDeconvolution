import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from scipy import stats  # noqa: E402

from bayspecdec import (  # noqa: E402
    BoxTransform,
    GammaPrior,
    UniformPrior,
    JaxModel,
    JaxParallelTempering,
    PTConfig,
    gaussian_log_likelihood,
    make_spectral_model,
    select_model_size,
)
from bayspecdec.evidence import estimate_evidence  # noqa: E402
from bayspecdec.tempering import warmup_schedule  # noqa: E402


def lorentz_basis(x, params):
    (mu,) = params
    return 0.5 / ((x[None, :] - mu[:, None]) ** 2 + 0.5)


def test_box_transform_roundtrip_and_jacobian():
    lower = np.array([0.0, 1.0, 0.5, -np.inf])
    upper = np.array([2.0, 3.0, np.inf, np.inf])
    tf = BoxTransform(lower, upper)
    z = jnp.array([-1.3, 0.4, 0.7, 2.0])

    theta = tf.to_theta(z)
    np.testing.assert_allclose(tf.to_z(theta), z, atol=1e-9)
    assert np.all(theta[:2] > lower[:2]) and np.all(theta[:2] < upper[:2])

    jac = jax.jacfwd(tf.to_theta)(z)
    np.testing.assert_allclose(tf.log_jac(z), np.sum(np.log(np.diag(jac))), atol=1e-9)


def test_gaussian_log_likelihood_matches_scipy():
    rng = np.random.default_rng(0)
    y, pred = rng.normal(size=50), rng.normal(size=50)
    ours = gaussian_log_likelihood(0.3)(jnp.asarray(y), jnp.asarray(pred))
    ref = stats.norm.logpdf(y, loc=pred, scale=np.sqrt(0.3)).sum()
    np.testing.assert_allclose(ours, ref, rtol=1e-10)


def _mlp_basis(seed=0, hidden=8):
    """Tiny neural basis: peak shape comes from an MLP of (x - mu)."""
    r = np.random.default_rng(seed)
    w1, b1 = r.normal(size=(hidden, 1)), r.normal(size=hidden)
    w2 = r.normal(size=(1, hidden)) / hidden

    def basis(x, params):
        (mu,) = params
        d = (x[None, :] - mu[:, None])[..., None]  # (K, n, 1)
        h = jnp.tanh(d @ w1.T + b1)
        return jnp.exp(-0.5 * d[..., 0] ** 2) + (h @ w2.T)[..., 0]

    return basis


@pytest.mark.parametrize("basis_fn", [lorentz_basis, _mlp_basis()])
def test_gradient_matches_finite_differences(basis_fn):
    x = np.linspace(0, 12, 60)
    y = np.random.default_rng(1).normal(size=60) * 0.01
    model = make_spectral_model(
        x, y, basis_fn, K=3, amplitude_prior=(0.2, 1.2),
        basis_priors=[(1.0, 11.0)], sigma2=1e-4,
    )
    z = jnp.asarray(np.random.default_rng(2).normal(size=model.ndim))

    def target(z):
        th = model.transform.to_theta(z)
        return model.log_likelihood(th) + model.transform.log_jac(z)

    g = jax.grad(target)(z)
    h = 1e-6
    fd = np.array(
        [
            (target(z.at[i].add(h)) - target(z.at[i].add(-h))) / (2 * h)
            for i in range(model.ndim)
        ]
    )
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-5 * np.abs(fd).max())


def test_swap_exchanges_accepted_pairs_only():
    model = _gauss_model(1)
    pt = JaxParallelTempering(model, np.array([0.0, 0.3, 0.6, 1.0]))
    z = jnp.arange(4.0)[:, None]
    lp = jnp.zeros(4)
    ll = jnp.array([3.0, 2.0, 1.0, 0.0])  # ll_l >= ll_{l+1}: log v >= 0, always accepted

    z0, ll0, _, active0, acc0 = pt._swap(jax.random.PRNGKey(0), z, ll, lp, 0)
    np.testing.assert_array_equal(np.asarray(acc0), [True, False, True])
    np.testing.assert_array_equal(np.asarray(z0[:, 0]), [1.0, 0.0, 3.0, 2.0])
    np.testing.assert_array_equal(np.asarray(ll0), [2.0, 3.0, 0.0, 1.0])

    z1, _, _, _, acc1 = pt._swap(jax.random.PRNGKey(0), z, ll, lp, 1)
    np.testing.assert_array_equal(np.asarray(acc1), [False, True, False])
    np.testing.assert_array_equal(np.asarray(z1[:, 0]), [0.0, 2.0, 1.0, 3.0])


def _gauss_model(d, half_width=5.0, std=0.5):
    """Likelihood prod N(theta_i; 0, std) under U(-a, a)^d prior: log Z = -d log(2a)."""
    tf = BoxTransform(np.full(d, -half_width), np.full(d, half_width))
    return JaxModel(
        ndim=d,
        log_likelihood=lambda th: jnp.sum(
            -0.5 * (th / std) ** 2 - jnp.log(std * jnp.sqrt(2 * jnp.pi))
        ),
        log_prior=lambda th: jnp.where(
            jnp.all(jnp.abs(th) <= half_width), -d * jnp.log(2 * half_width), -jnp.inf
        ),
        transform=tf,
        sample_prior=lambda key: jax.random.uniform(
            key, (d,), minval=-half_width, maxval=half_width
        ),
    )


def test_pt_recovers_analytic_evidence_and_posterior():
    d = 3
    model = _gauss_model(d)
    betas = np.concatenate([[0.0], 1.5 ** (np.arange(1, 24) - 23)])
    pt = JaxParallelTempering(model, betas, PTConfig(swap_every=5))
    result = pt.run(burn_in=500, samples=2000, seed=3)

    log_z = estimate_evidence(result).log_z
    assert abs(log_z - (-d * np.log(10.0))) < 0.3

    post = result.samples_by_temperature[-1]
    assert np.all(np.abs(post.mean(axis=0)) < 0.1)
    np.testing.assert_allclose(post.std(axis=0), 0.5, atol=0.07)
    assert 0.3 < result.within_acceptance[-1] < 1.0
    assert np.all(result.exchange_acceptance > 0.05)


def test_select_model_size_prefers_true_number_of_peaks():
    rng = np.random.default_rng(4)
    x = np.arange(0, 12, 0.1)
    truth = lorentz_basis(x, jnp.array([[3.0, 6.0, 10.0]]))
    y = np.array([1.0, 2.0, 2.0]) @ np.asarray(truth) + rng.normal(0, 0.01, x.size)

    def factory(K):
        return make_spectral_model(
            x, y, lorentz_basis, K, amplitude_prior=(0.2, 2.5),
            basis_priors=[(1.0, 11.0)], sigma2=1e-4,
        )

    betas = np.concatenate([[0.0], 1.5 ** (np.arange(1, 30) - 29)])
    runs = select_model_size(
        factory, [1, 3], betas, burn_in=800, samples=800, verbose=False,
        config=PTConfig(swap_every=5),
    )
    assert runs[1].evidence.log_z > runs[0].evidence.log_z


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs, ref", [
    ({"rate": 2.0}, stats.gamma(a=3.0, scale=0.5)),
    ({"scale": 0.04}, stats.gamma(a=3.0, scale=0.04)),
])
def test_gamma_prior_matches_scipy(kwargs, ref):
    prior = GammaPrior(3.0, **kwargs)
    x = np.array([0.01, 0.2, 1.0, 4.0])
    np.testing.assert_allclose(prior.log_prob(jnp.asarray(x)), ref.logpdf(x), rtol=1e-10)
    assert prior.log_prob(jnp.array([0.0, -1.0])).tolist() == [-np.inf, -np.inf]

    draws = np.asarray(prior.sample(jax.random.PRNGKey(0), (200_000,)))
    np.testing.assert_allclose(draws.mean(), ref.mean(), rtol=0.02)
    np.testing.assert_allclose(draws.var(), ref.var(), rtol=0.05)


def test_gamma_prior_gradient_is_finite_everywhere():
    prior = GammaPrior(5.0, rate=5.0)
    g = jax.grad(lambda x: jnp.sum(prior.log_prob(x)))(jnp.array([-1.0, 0.0, 0.5]))
    assert np.all(np.isfinite(g))


def test_prior_argument_validation():
    with pytest.raises(ValueError):
        GammaPrior(2.0)
    with pytest.raises(ValueError):
        GammaPrior(2.0, rate=1.0, scale=1.0)
    with pytest.raises(ValueError):
        GammaPrior(-1.0, rate=1.0)
    with pytest.raises(ValueError):
        UniformPrior(1.0, 1.0)


def test_uniform_prior_density_and_support():
    prior = UniformPrior(1.0, 5.0)
    lp = prior.log_prob(jnp.array([0.9, 1.0, 3.0, 5.0, 5.1]))
    np.testing.assert_allclose(lp, [-np.inf, np.log(0.25), np.log(0.25), np.log(0.25), -np.inf])
    draws = np.asarray(prior.sample(jax.random.PRNGKey(1), (1000,)))
    assert draws.min() >= 1.0 and draws.max() <= 5.0


def test_spectral_model_with_mixed_priors():
    x = np.linspace(0, 12, 40)
    y = np.zeros(40)
    model = make_spectral_model(
        x, y, lorentz_basis, K=2,
        amplitude_prior=GammaPrior(5.0, rate=5.0),
        basis_priors=[UniformPrior(1.0, 11.0)],
        sigma2=1e-2,
    )
    np.testing.assert_array_equal(model.transform.lower, [0, 0, 1, 1])
    assert np.isinf(model.transform.upper[:2]).all() and (model.transform.upper[2:] == 11).all()

    theta = model.sample_prior(jax.random.PRNGKey(0))
    assert theta.shape == (4,) and jnp.all(theta[:2] > 0) and jnp.all((theta[2:] >= 1) & (theta[2:] <= 11))
    assert np.isfinite(model.log_prior(theta))
    # log density in theta space = Gamma terms + uniform terms
    expected = GammaPrior(5.0, rate=5.0).log_prob(theta[:2]).sum() - 2 * np.log(10.0)
    np.testing.assert_allclose(model.log_prior(theta), expected, rtol=1e-12)

    z = model.transform.to_z(theta)
    g = jax.grad(lambda z: model.log_likelihood(model.transform.to_theta(z))
                 + model.log_prior(model.transform.to_theta(z))
                 + model.transform.log_jac(z))(z)
    assert np.all(np.isfinite(g))


def test_pt_with_gamma_prior_recovers_analytic_evidence():
    """Gamma(a, rate b) prior, L(theta) = exp(-c theta): Z = (b/(b+c))^a, posterior Gamma(a, b+c)."""
    a, b, c, d = 3.0, 2.0, 4.0, 2
    prior = GammaPrior(a, rate=b)
    model = JaxModel(
        ndim=d,
        log_likelihood=lambda th: -c * jnp.sum(th),
        log_prior=lambda th: jnp.sum(prior.log_prob(th)),
        transform=BoxTransform(np.zeros(d), np.full(d, np.inf)),
        sample_prior=lambda key: prior.sample(key, (d,)),
    )
    betas = np.concatenate([[0.0], 1.5 ** (np.arange(1, 20) - 19)])
    pt = JaxParallelTempering(model, betas, PTConfig(swap_every=5))
    result = pt.run(burn_in=500, samples=3000, seed=5)

    assert abs(estimate_evidence(result).log_z - d * a * np.log(b / (b + c))) < 0.15
    post = result.samples_by_temperature[-1]
    np.testing.assert_allclose(post.mean(axis=0), a / (b + c), rtol=0.1)


# ---------------------------------------------------------------------------
# Mass-matrix adaptation
# ---------------------------------------------------------------------------


def test_warmup_schedule_layout():
    cfg = PTConfig()
    acc, end = warmup_schedule(800, cfg)
    assert list(np.flatnonzero(end) + 1) == [100, 150, 250, 750]  # doubling, last absorbs the rest
    assert acc[:75].sum() == 0 and acc[75:750].all() and acc[750:].sum() == 0

    acc, end = warmup_schedule(100, cfg)  # too short for defaults: 15% / 10% split
    assert end.sum() >= 1 and not acc[-10:].any() and not acc[:15].any()

    assert not any(a.any() for a in warmup_schedule(10, cfg))
    assert not any(a.any() for a in warmup_schedule(800, PTConfig(adapt_mass=False)))


def _skewed_gauss_model(stds, half_width=50.0):
    stds = np.asarray(stds)
    d = stds.size
    return JaxModel(
        ndim=d,
        log_likelihood=lambda th: jnp.sum(-0.5 * (th / stds) ** 2 - jnp.log(stds * jnp.sqrt(2 * jnp.pi))),
        log_prior=lambda th: jnp.where(
            jnp.all(jnp.abs(th) <= half_width), -d * jnp.log(2 * half_width), -jnp.inf
        ),
        transform=BoxTransform(np.full(d, -half_width), np.full(d, half_width)),
        sample_prior=lambda key: jax.random.uniform(
            key, (d,), minval=-half_width, maxval=half_width
        ),
    )


def test_mass_adaptation_learns_scales_and_improves_mixing():
    stds = np.array([0.05, 0.5, 5.0])  # two orders of magnitude apart
    model = _skewed_gauss_model(stds)
    betas = np.concatenate([[0.0], 1.5 ** (np.arange(1, 26) - 25)])

    outcome = {}
    for adapt in (False, True):
        pt = JaxParallelTempering(model, betas, PTConfig(adapt_mass=adapt, swap_every=5))
        res = pt.run(burn_in=800, samples=2000, seed=1)
        x = res.samples_by_temperature[-1][:, 2]
        x = x - x.mean()
        outcome[adapt] = (pt, res, (x[1:] * x[:-1]).mean() / x.var())

    pt0, _, ac0 = outcome[False]
    pt1, res1, ac1 = outcome[True]

    np.testing.assert_array_equal(pt0.final_inverse_mass[-1], 1.0)  # untouched when off
    inv_m = pt1.final_inverse_mass[-1]
    # theta std ratios 10 -> variance ratio ~100 (z ~ linear in theta near the centre)
    assert 50 < inv_m[2] / inv_m[1] < 200
    assert inv_m[1] > inv_m[0]

    assert pt1.final_step_sizes[-1] > 10 * pt0.final_step_sizes[-1]
    assert ac1 < ac0 - 0.5
    np.testing.assert_allclose(res1.samples_by_temperature[-1].std(axis=0), stds, rtol=0.2)
    assert abs(estimate_evidence(res1).log_z - (-3 * np.log(100.0))) < 0.4


def test_user_supplied_initial_inverse_mass_is_used_when_adaptation_off():
    model = _gauss_model(2)
    pt = JaxParallelTempering(
        model, np.array([0.0, 0.5, 1.0]), PTConfig(adapt_mass=False), inverse_mass=np.array([0.3, 2.0])
    )
    pt.run(burn_in=30, samples=10)
    np.testing.assert_allclose(pt.final_inverse_mass, [[0.3, 2.0]] * 3)


# -- likelihoods ---------------------------------------------------------------------------


def test_poisson_log_likelihood_matches_scipy_and_has_finite_gradient():
    from bayspecdec import poisson_log_likelihood

    rng = np.random.default_rng(0)
    lam = rng.uniform(0.5, 20.0, size=40)
    y = rng.poisson(lam).astype(float)
    ll = poisson_log_likelihood()
    np.testing.assert_allclose(ll(jnp.asarray(y), jnp.asarray(lam)), stats.poisson.logpmf(y, lam).sum())
    grad = jax.grad(lambda p: ll(jnp.asarray(y), p))(jnp.asarray(lam - 5.0))  # some lam < 0
    assert np.all(np.isfinite(grad))


@pytest.mark.parametrize("sigma2, lam_max", [(1e-4, 30.0), (0.5, 10.0), (4.0, 30.0)])
def test_poisson_gaussian_matches_brute_force_sum(sigma2, lam_max):
    from bayspecdec import poisson_gaussian_log_likelihood

    rng = np.random.default_rng(1)
    lam = np.concatenate([[0.0, 1e-3], rng.uniform(0.0, lam_max, size=30)])
    y = rng.poisson(lam) + rng.normal(0.0, np.sqrt(sigma2), size=lam.size)
    ours = poisson_gaussian_log_likelihood(sigma2, half_width=25)(jnp.asarray(y), jnp.asarray(lam))
    k = np.arange(0, 200)[:, None]
    terms = stats.poisson.logpmf(k, lam[None, :]) + stats.norm.logpdf(y[None, :], loc=k, scale=np.sqrt(sigma2))
    ref = np.sum(np.logaddexp.reduce(terms, axis=0))
    np.testing.assert_allclose(ours, ref, rtol=1e-8)


def test_poisson_gaussian_gradient_matches_finite_differences():
    from bayspecdec import poisson_gaussian_log_likelihood

    rng = np.random.default_rng(2)
    lam = jnp.asarray(rng.uniform(1.0, 8.0, size=10))
    y = jnp.asarray(rng.poisson(np.asarray(lam)) + rng.normal(0, 0.3, size=10))
    f = lambda p: poisson_gaussian_log_likelihood(0.09)(y, p)
    g = jax.grad(f)(lam)
    h = 1e-6
    fd = np.array([(f(lam.at[i].add(h)) - f(lam.at[i].add(-h))) / (2 * h) for i in range(10)])
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


def test_poisson_gaussian_rejects_bad_arguments():
    from bayspecdec import poisson_gaussian_log_likelihood

    with pytest.raises(ValueError):
        poisson_gaussian_log_likelihood(0.0)
    with pytest.raises(ValueError):
        poisson_gaussian_log_likelihood(1.0, half_width=0)


# -- priors and basis ----------------------------------------------------------------------


def test_normal_prior_matches_scipy_and_samples():
    from bayspecdec import NormalPrior

    p = NormalPrior(1.5, 5.0)
    x = np.linspace(-10, 10, 21)
    np.testing.assert_allclose(p.log_prob(jnp.asarray(x)), stats.norm.logpdf(x, 1.5, 5.0))
    draws = np.asarray(p.sample(jax.random.PRNGKey(0), (20000,)))
    assert abs(draws.mean() - 1.5) < 0.15 and abs(draws.std() - 5.0) < 0.15
    with pytest.raises(ValueError):
        NormalPrior(0.0, 0.0)


def test_basis_functions_match_closed_form():
    from bayspecdec import gaussian_basis, lorentzian_basis

    x = np.linspace(0, 3, 31)
    mu, w = np.array([1.0, 2.0]), np.array([50.0, 100.0])
    params = jnp.asarray(np.stack([mu, w]))
    np.testing.assert_allclose(
        gaussian_basis(jnp.asarray(x), params), np.exp(-0.5 * w[:, None] * (x[None] - mu[:, None]) ** 2)
    )
    np.testing.assert_allclose(
        lorentzian_basis(jnp.asarray(x), params), 1 / (1 + ((x[None] - mu[:, None]) / w[:, None]) ** 2)
    )


def test_paper_synthetic_model_runs_with_normal_and_gamma_priors():
    from bayspecdec.data import make_paper_like_synthetic_data
    from bayspecdec import gaussian_basis, paper_synthetic_priors

    x, y, _ = make_paper_like_synthetic_data()
    amp, basis = paper_synthetic_priors()
    model = make_spectral_model(x, y, gaussian_basis, 3, amp, basis, sigma2=0.01)
    theta = model.sample_prior(jax.random.PRNGKey(0))
    assert np.isfinite(model.log_likelihood(theta)) and np.isfinite(model.log_prior(theta))
    z = model.transform.to_z(theta)
    assert np.all(np.isfinite(jax.grad(lambda z: model.log_posterior(model.transform.to_theta(z)))(z)))


# -- NUTS and random-walk kernels ------------------------------------------------------------

_BETAS = np.concatenate([[0.0], 1.5 ** (np.arange(1, 24) - 23)])


def test_nuts_recovers_analytic_evidence_and_posterior():
    d = 3
    pt = JaxParallelTempering(_gauss_model(d), _BETAS, PTConfig(kernel="nuts", swap_every=5))
    result = pt.run(burn_in=500, samples=1500, seed=3)

    assert abs(estimate_evidence(result).log_z - (-d * np.log(10.0))) < 0.3
    post = result.samples_by_temperature[-1]
    assert np.all(np.abs(post.mean(axis=0)) < 0.1)
    np.testing.assert_allclose(post.std(axis=0), 0.5, atol=0.07)
    assert np.all(result.exchange_acceptance > 0.05)


def test_nuts_samples_correlated_gaussian_with_correct_covariance():
    # Posterior N(0, S) with strong correlation; exercises the U-turn logic beyond a diagonal target.
    S = np.array([[1.0, 0.9], [0.9, 1.0]])
    P = jnp.asarray(np.linalg.inv(S))
    model = JaxModel(
        ndim=2,
        log_likelihood=lambda th: -0.5 * th @ P @ th,
        log_prior=lambda th: jnp.where(jnp.all(jnp.abs(th) <= 20.0), 0.0, -jnp.inf),
        transform=BoxTransform(np.full(2, -20.0), np.full(2, 20.0)),
        sample_prior=lambda key: jax.random.uniform(key, (2,), minval=-5.0, maxval=5.0),
    )
    pt = JaxParallelTempering(model, [0.5, 1.0], PTConfig(kernel="nuts", swap_every=5))
    post = pt.run(burn_in=500, samples=3000, seed=0).samples_by_temperature[-1]
    np.testing.assert_allclose(np.cov(post.T), S, atol=0.15)
    np.testing.assert_allclose(post.mean(axis=0), 0.0, atol=0.15)


def test_nuts_depth_is_bounded_and_runs_with_adaptation_off():
    pt = JaxParallelTempering(
        _gauss_model(2), _BETAS[-4:], PTConfig(kernel="nuts", max_tree_depth=2, adapt_mass=False)
    )
    result = pt.run(burn_in=50, samples=100, seed=1)
    assert np.all(np.isfinite(result.log_likelihood_trace_by_temperature[-1]))


def test_rwm_recovers_analytic_evidence_and_posterior():
    d = 2
    pt = JaxParallelTempering(_gauss_model(d), _BETAS, PTConfig(kernel="rwm", swap_every=2))
    result = pt.run(burn_in=2000, samples=10000, seed=4)

    assert abs(estimate_evidence(result).log_z - (-d * np.log(10.0))) < 0.4
    post = result.samples_by_temperature[-1]
    np.testing.assert_allclose(post.std(axis=0), 0.5, atol=0.1)
    assert 0.1 < result.within_acceptance[-1] < 0.6  # dual averaging aims at 0.234


def test_kernel_config_validation_and_defaults():
    assert PTConfig().target_accept == 0.8
    assert PTConfig(kernel="nuts").target_accept == 0.8
    assert PTConfig(kernel="rwm").target_accept == 0.234
    assert PTConfig(kernel="rwm", target_accept=0.3).target_accept == 0.3
    with pytest.raises(ValueError):
        PTConfig(kernel="gibbs")
    with pytest.raises(ValueError):
        PTConfig(kernel="nuts", max_tree_depth=0)
    with pytest.raises(ValueError):
        PTConfig(target_accept=1.5)


def test_spectral_model_with_poisson_gaussian_likelihood_runs_under_nuts():
    from bayspecdec import poisson_gaussian_log_likelihood

    x = np.linspace(0, 6, 40)
    truth = 10.0 * 0.5 / ((x - 3.0) ** 2 + 0.5)
    rng = np.random.default_rng(0)
    y = rng.poisson(truth) + rng.normal(0, 0.1, size=x.size)
    model = make_spectral_model(
        x, y, lorentz_basis, 1, (1.0, 30.0), [(1.0, 5.0)],
        likelihood=poisson_gaussian_log_likelihood(0.01),
    )
    pt = JaxParallelTempering(model, 1.5 ** (np.arange(-8, 1)), PTConfig(kernel="nuts", swap_every=5))
    post = pt.run(burn_in=300, samples=400, seed=0).samples_by_temperature[-1]
    assert abs(post[:, 1].mean() - 3.0) < 0.2  # peak centre recovered


def test_poisson_gaussian_with_gain_matches_direct_sum():
    from scipy.special import logsumexp
    from bayspecdec import poisson_gaussian_log_likelihood

    rng = np.random.default_rng(3)
    gain, sigma2 = 4.0, 25.0
    mean = rng.uniform(5.0, 400.0, size=20)  # prediction, in counts
    y = gain * rng.poisson(mean / gain) + rng.normal(0, np.sqrt(sigma2), 20)
    k = np.arange(0, 400)[None, :]
    ref = logsumexp(
        stats.poisson.logpmf(k, (mean / gain)[:, None]) + stats.norm.logpdf(y[:, None], gain * k, np.sqrt(sigma2)),
        axis=1,
    ).sum()
    ours = poisson_gaussian_log_likelihood(sigma2, half_width=60, gain=gain)(jnp.asarray(y), jnp.asarray(mean))
    np.testing.assert_allclose(ours, ref, rtol=1e-9)


def test_poisson_gaussian_gives_clear_error_when_passed_uncalled():
    from bayspecdec import poisson_gaussian_log_likelihood

    with pytest.raises(TypeError, match="returns the likelihood"):
        poisson_gaussian_log_likelihood(jnp.ones(3))
    with pytest.raises(ValueError):
        poisson_gaussian_log_likelihood(1.0, gain=0.0)


def test_heteroscedastic_gaussian_approximates_poisson_gaussian_at_large_counts():
    from bayspecdec import (
        heteroscedastic_gaussian_log_likelihood,
        poisson_gaussian_log_likelihood,
    )

    rng = np.random.default_rng(4)
    gain, read_var = 5.0, 900.0
    mean = rng.uniform(2e3, 2e5, size=30)
    y = gain * rng.poisson(mean / gain) + rng.normal(0, 30.0, 30)
    exact = poisson_gaussian_log_likelihood(read_var, half_width=400, gain=gain)(jnp.asarray(y), jnp.asarray(mean))
    approx = heteroscedastic_gaussian_log_likelihood(read_var, gain)(jnp.asarray(y), jnp.asarray(mean))
    assert abs(float(exact - approx)) < 0.05 * 30  # < 0.05 nats per bin
    grad = jax.grad(lambda p: heteroscedastic_gaussian_log_likelihood(read_var, gain)(jnp.asarray(y), p))(jnp.asarray(mean))
    assert np.all(np.isfinite(grad))


# -- sampled noise hyperparameter ------------------------------------------------------------


def test_fermi_dirac_prior_is_normalised_and_sampler_matches():
    from scipy.integrate import quad
    from bayspecdec import FermiDiracPrior

    p = FermiDiracPrior(mu=1.0, temperature=0.2)
    total = quad(lambda x: float(np.exp(p.log_prob(jnp.asarray(x)))), 0.0, 40.0, points=[1.0])[0]
    np.testing.assert_allclose(total, 1.0, rtol=1e-6)
    s = np.asarray(p.sample(jax.random.PRNGKey(0), (100_000,)))
    f = lambda t: 1.0 / (1.0 + np.exp((t - 1.0) / 0.2))
    norm = quad(f, 0.0, 40.0)[0]
    for q in (0.5, 1.0, 1.5):
        np.testing.assert_allclose((s < q).mean(), quad(f, 0.0, q)[0] / norm, atol=0.01)


def _noisy_lorentz_data(sigma2, n=200, seed=0):
    x = np.linspace(0.0, 6.0, n)
    clean = 2.0 * 0.5 / ((x - 3.0) ** 2 + 0.5)
    y = clean + np.random.default_rng(seed).normal(0.0, np.sqrt(sigma2), n)
    return x, y


def test_sampled_noise_layout_split_and_validation():
    from bayspecdec import log_scale_prior

    x, y = _noisy_lorentz_data(0.01, n=20)
    m = make_spectral_model(
        x, y, lorentz_basis, 2, (0.5, 5.0), [(1.0, 5.0)], noise_prior=log_scale_prior(1e-6, 1.0)
    )
    assert m.ndim == 2 * 2 + 1 and m.n_noise == 1
    theta = m.sample_prior(jax.random.PRNGKey(0))
    phys, noise = m.split(theta)
    assert phys.shape == (4,) and noise.shape == (1,)
    assert np.log(1e-6) <= float(noise[0]) <= 0.0
    assert m.predict(theta).shape == (20,)
    assert m.transform.lower[-1] == np.log(1e-6) and m.transform.upper[-1] == 0.0
    # the sigma^2 term is part of the density: ll must vary with s
    assert float(m.log_likelihood(theta.at[-1].set(-8.0))) != float(m.log_likelihood(theta.at[-1].set(-2.0)))

    with pytest.raises(ValueError):
        make_spectral_model(x, y, lorentz_basis, 1, (0.5, 5.0), [(1.0, 5.0)], sigma2=0.1, noise_prior=log_scale_prior(1e-6, 1.0))
    with pytest.raises(ValueError):
        make_spectral_model(x, y, lorentz_basis, 1, (0.5, 5.0), [(1.0, 5.0)], noise_likelihood=lambda y, p, s: 0.0)
    with pytest.raises(ValueError):
        make_spectral_model(x, y, lorentz_basis, 1, (0.5, 5.0), [(1.0, 5.0)])


def test_fixed_noise_models_are_unchanged_by_the_noise_machinery():
    x, y = _noisy_lorentz_data(0.01, n=20)
    m = make_spectral_model(x, y, lorentz_basis, 1, (0.5, 5.0), [(1.0, 5.0)], sigma2=0.01)
    assert m.n_noise == 0 and m.ndim == 2
    phys, noise = m.split(jnp.array([2.0, 3.0]))
    assert phys.shape == (2,) and noise.shape == (0,)


@pytest.mark.parametrize("kernel", ["hmc", "nuts"])
def test_sampled_sigma2_is_recovered_and_amplitude_agrees_with_fixed_noise(kernel):
    from bayspecdec import log_scale_prior

    true_s2 = 0.04
    x, y = _noisy_lorentz_data(true_s2, n=300)
    args = (x, y, lorentz_basis, 1, (0.5, 5.0), [(1.0, 5.0)])
    betas = 1.5 ** (np.arange(-12, 1))
    cfg = PTConfig(kernel=kernel, swap_every=5)

    free = make_spectral_model(*args, noise_prior=log_scale_prior(1e-6, 1.0))
    r_free = JaxParallelTempering(free, betas, cfg).run(burn_in=600, samples=1500, seed=1)
    post = r_free.samples_by_temperature[-1]
    s2 = np.exp(post[:, -1])
    assert abs(s2.mean() - true_s2) < 0.25 * true_s2  # ~ sigma^2 * n / (n - dof) +- sqrt(2/n)

    fixed = make_spectral_model(*args, sigma2=true_s2)
    r_fix = JaxParallelTempering(fixed, betas, cfg).run(burn_in=600, samples=1500, seed=1)
    np.testing.assert_allclose(post[:, :2].mean(axis=0), r_fix.samples_by_temperature[-1].mean(axis=0), atol=0.1)


def test_sampled_noise_gives_comparable_evidence_across_K():
    from bayspecdec import log_scale_prior

    x, y = _noisy_lorentz_data(0.01, n=150)

    def factory(K):
        return make_spectral_model(
            x, y, lorentz_basis, K, (0.2, 5.0), [(0.5, 5.5)], noise_prior=log_scale_prior(1e-6, 1.0)
        )

    runs = select_model_size(
        factory, [1, 2], betas=1.5 ** (np.arange(-20, 1)), burn_in=600, samples=800,
        config=PTConfig(swap_every=5), verbose=False,
    )
    assert np.all(np.isfinite([r.stochastic_complexity for r in runs]))
    assert min(runs, key=lambda r: r.stochastic_complexity).K == 1  # data hold exactly one peak


# --- background --------------------------------------------------------------------------------


def test_arctan_step_background_matches_paper_formula():
    from bayspecdec import arctan_step_background

    x = np.linspace(520.0, 590.0, 50)
    H, E0, G, A, dE, w = 0.8, 535.0, 2.0, 0.6, 3.0, 4.0
    ref = H * (0.5 + np.arctan((x - E0) / (G / 2)) / np.pi) + A * np.exp(
        -4 * np.log(2) * ((x - (E0 + dE)) / w) ** 2
    )
    got = arctan_step_background(jnp.asarray(x), jnp.array([H, E0, G, A, dE, w]))
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-12)


def test_polynomial_background_matches_numpy():
    from bayspecdec import polynomial_background

    x = np.linspace(-1.0, 2.0, 11)
    c = np.array([0.5, -1.0, 2.0])
    np.testing.assert_allclose(
        np.asarray(polynomial_background(2)(jnp.asarray(x), jnp.asarray(c))), np.polynomial.polynomial.polyval(x, c)
    )


def test_background_layout_predict_prior_and_gradient():
    from bayspecdec import log_scale_prior, polynomial_background

    x, y = _noisy_lorentz_data(0.01, n=20)
    bg = polynomial_background(1)
    m = make_spectral_model(
        x, y, lorentz_basis, 2, (0.5, 5.0), [(1.0, 5.0)],
        background_fn=bg, background_priors=[(-1.0, 1.0), (-0.5, 0.5)],
        noise_prior=log_scale_prior(1e-6, 1.0),
    )
    assert m.ndim == 2 * 2 + 2 + 1 and m.n_background == 2 and m.n_noise == 1
    theta = m.sample_prior(jax.random.PRNGKey(0))
    peaks, bgp, noise = m.split_background(theta)
    assert peaks.shape == (4,) and bgp.shape == (2,) and noise.shape == (1,)
    assert -1.0 <= float(bgp[0]) <= 1.0 and -0.5 <= float(bgp[1]) <= 0.5
    assert m.split(theta)[0].shape == (6,)
    np.testing.assert_allclose(m.transform.lower[4:6], [-1.0, -0.5])

    no_bg = make_spectral_model(
        x, y, lorentz_basis, 2, (0.5, 5.0), [(1.0, 5.0)], noise_prior=log_scale_prior(1e-6, 1.0)
    )
    theta_nb = jnp.concatenate([peaks, noise])
    np.testing.assert_allclose(
        np.asarray(m.predict(theta) - no_bg.predict(theta_nb)), np.asarray(bg(jnp.asarray(x), bgp)), atol=1e-12
    )
    # a background parameter outside its prior has zero density; inside, the gradient is finite
    assert not np.isfinite(float(m.log_prior(theta.at[4].set(2.0))))
    assert np.all(np.isfinite(np.asarray(jax.grad(m.log_posterior)(theta))))


def test_background_argument_validation():
    from bayspecdec import arctan_step_background, constant_background

    x, y = _noisy_lorentz_data(0.01, n=20)
    args = (x, y, lorentz_basis, 1, (0.5, 5.0), [(1.0, 5.0)])
    with pytest.raises(ValueError):
        make_spectral_model(*args, sigma2=0.01, background_fn=constant_background)
    with pytest.raises(ValueError):
        make_spectral_model(*args, sigma2=0.01, background_priors=[(0.0, 1.0)])
    with pytest.raises(ValueError):  # arctan step needs six priors
        make_spectral_model(*args, sigma2=0.01, background_fn=arctan_step_background, background_priors=[(0.0, 1.0)])


def test_pt_recovers_step_height_alongside_a_peak():
    from bayspecdec import arctan_step_background

    rng = np.random.default_rng(3)
    sigma2 = 0.0025
    x = np.linspace(0.0, 10.0, 200)
    truth = np.array([1.0, 5.0, 0.3, 0.0, 0.0, 1.0])  # H, E0, Gamma, A, dE, omega (white line off)
    peak = 0.8 * np.exp(-0.5 * 8.0 * (x - 3.0) ** 2)
    y = peak + np.asarray(arctan_step_background(jnp.asarray(x), jnp.asarray(truth))) + rng.normal(0, np.sqrt(sigma2), x.size)

    def gauss(xx, p):
        mu, b = p
        return jnp.exp(-0.5 * b[:, None] * (xx[None, :] - mu[:, None]) ** 2)

    m = make_spectral_model(
        x, y, gauss, 1, (0.1, 3.0), [(1.0, 9.0), (1.0, 30.0)],
        background_fn=arctan_step_background,
        background_priors=[(0.0, 2.0), (4.0, 6.0), (0.1, 1.0), (0.0, 0.05), (-0.1, 0.1), (0.5, 2.0)],
        sigma2=sigma2,
    )
    betas = 1.5 ** (np.arange(-14, 1))
    res = JaxParallelTempering(m, betas, PTConfig(swap_every=5)).run(burn_in=800, samples=1500, seed=0)
    post = res.samples_by_temperature[-1]
    H = post[:, 3 * 1]  # first background column (after a, mu, b for K=1)
    assert abs(H.mean() - 1.0) < 0.05
    assert abs(post[:, 1].mean() - 3.0) < 0.1  # peak centre unaffected by the step
