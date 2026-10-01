import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from scipy import stats  # noqa: E402

from bayspecdec.jax_backend import (  # noqa: E402
    BoxTransform,
    JaxModel,
    JaxParallelTempering,
    PTConfig,
    gaussian_log_likelihood,
    make_spectral_model,
    select_model_size,
)
from bayspecdec.evidence import estimate_evidence  # noqa: E402


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
        x, y, basis_fn, K=3, amplitude_bounds=(0.2, 1.2),
        basis_param_bounds=[(1.0, 11.0)], sigma2=1e-4,
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
            x, y, lorentz_basis, K, amplitude_bounds=(0.2, 2.5),
            basis_param_bounds=[(1.0, 11.0)], sigma2=1e-4,
        )

    betas = np.concatenate([[0.0], 1.5 ** (np.arange(1, 30) - 29)])
    runs = select_model_size(
        factory, [1, 3], betas, burn_in=800, samples=800, verbose=False,
        config=PTConfig(swap_every=5),
    )
    assert runs[1].evidence.log_z > runs[0].evidence.log_z
