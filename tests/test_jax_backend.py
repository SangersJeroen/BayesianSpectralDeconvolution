import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from scipy import stats  # noqa: E402

from bayspecdec.jax_backend import (  # noqa: E402
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
from bayspecdec.jax_backend.tempering import warmup_schedule  # noqa: E402


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

    log_z = estimate_evidence(None, result).log_z
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

    assert abs(estimate_evidence(None, result).log_z - d * a * np.log(b / (b + c))) < 0.15
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
    assert abs(estimate_evidence(None, res1).log_z - (-3 * np.log(100.0))) < 0.4


def test_user_supplied_initial_inverse_mass_is_used_when_adaptation_off():
    model = _gauss_model(2)
    pt = JaxParallelTempering(
        model, np.array([0.0, 0.5, 1.0]), PTConfig(adapt_mass=False), inverse_mass=np.array([0.3, 2.0])
    )
    pt.run(burn_in=30, samples=10)
    np.testing.assert_allclose(pt.final_inverse_mass, [[0.3, 2.0]] * 3)
