import jax
import jax.numpy as jnp
import optax
import pytest

from stam_optimizer import STAM, STAMLite, stam, stam_lite


@pytest.fixture
def params():
    return {"layer": {"kernel": jnp.ones((3, 2)), "bias": jnp.zeros((2,))}}


def leaves(tree):
    return jax.tree_util.tree_leaves(tree)


@pytest.mark.parametrize("optimizer_cls", [STAM, STAMLite])
def test_update_advances_state_and_is_finite(optimizer_cls, params):
    optimizer = optimizer_cls()
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, state = optimizer.update(grads, optimizer.init(params), params)
    assert int(state.count) == 1
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves((updates, state)))


@pytest.mark.parametrize("optimizer_cls", [STAM, STAMLite])
@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_nonfinite_gradients_are_not_silently_sanitized(optimizer_cls, params, bad_value):
    optimizer = optimizer_cls(weight_decay=0.0)
    grads = jax.tree_util.tree_map(lambda p: jnp.full_like(p, bad_value), params)
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    assert any(bool(jnp.any(~jnp.isfinite(x))) for x in leaves(updates))


@pytest.mark.parametrize("optimizer_cls", [STAM, STAMLite])
def test_decoupled_weight_decay_matches_contract(optimizer_cls, params):
    lr, decay = 0.1, 0.2
    optimizer = optimizer_cls(learning_rate=lr, weight_decay=decay)
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    assert jnp.allclose(updates["layer"]["kernel"], lr * decay * params["layer"]["kernel"])


@pytest.mark.parametrize("optimizer_cls,kwargs", [
    (STAM, {"learning_rate": 0}), (STAM, {"eps": 0}),
    (STAM, {"weight_decay": -1}), (STAM, {"tau_decay": 1}),
    (STAM, {"fallback_threshold": 0}), (STAMLite, {"learning_rate": -1}),
    (STAMLite, {"eps": 0}), (STAMLite, {"weight_decay": -1}),
    (STAMLite, {"moment_decay": 1}), (STAMLite, {"beta1_update_interval": 0}),
])
def test_invalid_hyperparameters_fail_fast(optimizer_cls, kwargs):
    with pytest.raises(ValueError):
        optimizer_cls(**kwargs)


@pytest.mark.parametrize("factory", [stam, stam_lite])
def test_optax_adapter_matches_direct_update_sign(factory, params):
    grads = jax.tree_util.tree_map(lambda p: jnp.full_like(p, 0.25), params)
    direct_cls = STAM if factory is stam else STAMLite
    direct = direct_cls(weight_decay=0.01)
    direct_updates, _ = direct.update(grads, direct.init(params), params)
    tx = factory(weight_decay=0.01)
    optax_updates, _ = tx.update(grads, tx.init(params), params)
    for expected, actual in zip(leaves(direct_updates), leaves(optax_updates)):
        assert jnp.allclose(actual, -expected)


def test_optax_adapter_composes_with_apply_updates(params):
    tx = stam(learning_rate=0.01, weight_decay=0.0)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, tx.init(params), params)
    result = optax.apply_updates(params, updates)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves(result))


def test_update_is_jittable(params):
    optimizer = STAMLite(weight_decay=0.0)
    state = optimizer.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    step = jax.jit(lambda g, s, p: optimizer.update(g, s, p))
    updates, next_state = step(grads, state, params)
    assert int(next_state.count) == 1
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves(updates))


@pytest.mark.parametrize("optimizer_cls", [STAM, STAMLite])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_hyperparameters_fail_fast(optimizer_cls, value):
    with pytest.raises(ValueError, match="finite"):
        optimizer_cls(learning_rate=value)


def test_benchmark_savers_accept_filename_without_directory(tmp_path, monkeypatch):
    from stam_optimizer.benchmarks import utils

    monkeypatch.chdir(tmp_path)
    utils.save_results([], "bare.json")
    assert (tmp_path / "bare.json").is_file()


def test_fair_timing_rejects_empty_measurement(monkeypatch, tmp_path):
    from stam_optimizer.benchmarks.fair_timing import run_fair_timing, time_optimizer

    with pytest.raises(ValueError, match="timed_steps"):
        time_optimizer("stam_lite", 0, 0, 0, 1e-3, 0.0)
    with pytest.raises(ValueError, match="must not be empty"):
        run_fair_timing([], [0], 0, 1, str(tmp_path / "out.json"))


@pytest.mark.parametrize("kwargs", [
    {"b1_base": float("nan")}, {"b2": float("inf")},
    {"adapt_strength": float("nan")}, {"tau_decay": float("inf")},
])
def test_stam_rejects_nonfinite_hyperparameters(kwargs):
    with pytest.raises(ValueError, match="finite"):
        STAM(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"b1_base": float("nan")}, {"adapt_strength": float("inf")},
    {"moment_decay": float("nan")}, {"beta1_update_interval": 1.5},
])
def test_stam_lite_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        STAMLite(**kwargs)


def test_benchmark_helpers_validate_inputs():
    from stam_optimizer.benchmarks.utils import loss_stability, timing_summary

    with pytest.raises(ValueError, match="tail_fraction"):
        loss_stability([1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="warmup_steps"):
        timing_summary([0.1], -1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        timing_summary([float("nan")])
