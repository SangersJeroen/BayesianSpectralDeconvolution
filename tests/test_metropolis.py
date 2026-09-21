import numpy as np
import pytest
from bayspecdec.samplers.metropolis import RandomWalkMetropolis, MetropolisState, metropolis_kernel_factory


class ExponentialParameterization:
    """1D positive parameter theta = exp(z)."""
    ndim = 1
    K = 1

    def __init__(self):
        self.jac_call_count = 0

    def unpack(self, theta):
        return (theta,)

    def pack(self, structured):
        return structured[0]

    def to_z(self, theta):
        return np.log(theta)

    def from_z(self, z):
        return np.exp(z)

    def log_jacobian(self, theta):
        self.jac_call_count += 1
        return float(np.log(theta[0]))


class DummyExpModel:
    n = 1

    def __init__(self, param=None):
        self.parameterization = param if param is not None else ExponentialParameterization()

    def log_tempered_target(self, theta, beta):
        # Target: p(theta) = exp(-theta) for theta > 0
        val = float(theta[0])
        if val <= 0:
            return -np.inf
        return -val

    def energy(self, theta):
        return 0.0


def test_metropolis_uses_log_jacobian():
    param = ExponentialParameterization()
    model = DummyExpModel(param=param)
    rng = np.random.default_rng(42)
    sampler = RandomWalkMetropolis(model, beta=1.0, rng=rng, proposal_scales=np.array([0.1]))

    state = MetropolisState(theta=np.array([1.0]), log_target=-1.0, energy=0.0)
    new_state = sampler.step(state)

    assert param.jac_call_count >= 2
    assert new_state.attempted == 1


def test_metropolis_disable_log_jacobian():
    param = ExponentialParameterization()
    model = DummyExpModel(param=param)
    rng = np.random.default_rng(42)
    sampler = RandomWalkMetropolis(
        model, beta=1.0, rng=rng, proposal_scales=np.array([0.1]), use_log_jacobian=False
    )

    state = MetropolisState(theta=np.array([1.0]), log_target=-1.0, energy=0.0)
    _ = sampler.step(state)

    assert param.jac_call_count == 0


def test_metropolis_kernel_factory_flag():
    param = ExponentialParameterization()
    model = DummyExpModel(param=param)
    rng = np.random.default_rng(42)

    s1 = metropolis_kernel_factory(model, 1.0, rng, use_log_jacobian=True)
    assert s1.use_log_jacobian is True

    s2 = metropolis_kernel_factory(model, 1.0, rng, use_log_jacobian=False)
    assert s2.use_log_jacobian is False
