from typing import Any, Callable, NamedTuple

import jax
import optax

from .core.stam import STAM
from .core.stam_lite import STAMLite


class STAMOptaxState(NamedTuple):
    inner_state: Any


def _as_optax_transform(make_optimizer: Callable[[], Any]) -> optax.GradientTransformation:
    def init_fn(params: Any) -> STAMOptaxState:
        optimizer = make_optimizer()
        return STAMOptaxState(inner_state=optimizer.init(params))

    def update_fn(updates: Any, state: STAMOptaxState, params: Any = None):
        if params is None:
            raise ValueError("params must be provided for STAM optax transformations")
        optimizer = make_optimizer()
        descent_updates, inner_state = optimizer.update(updates, state.inner_state, params)
        optax_updates = jax.tree_util.tree_map(lambda u: -u, descent_updates)
        return optax_updates, STAMOptaxState(inner_state=inner_state)

    return optax.GradientTransformation(init_fn, update_fn)


def stam(
    learning_rate: float = 1e-3,
    b1_base: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    adapt_strength: float = 0.2,
    tau_decay: float = 0.99,
    fallback_threshold: float = 100.0,
) -> optax.GradientTransformation:
    def make_optimizer() -> STAM:
        return STAM(
            learning_rate=learning_rate,
            b1_base=b1_base,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            adapt_strength=adapt_strength,
            tau_decay=tau_decay,
            fallback_threshold=fallback_threshold,
        )

    return _as_optax_transform(make_optimizer)


def stam_lite(
    learning_rate: float = 1e-3,
    b1_base: float = 0.9,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    adapt_strength: float = 0.2,
    moment_decay: float = 0.99,
    beta1_update_interval: int = 5,
    state_dtype: Any = None,
) -> optax.GradientTransformation:
    def make_optimizer() -> STAMLite:
        kwargs = {
            "learning_rate": learning_rate,
            "b1_base": b1_base,
            "eps": eps,
            "weight_decay": weight_decay,
            "adapt_strength": adapt_strength,
            "moment_decay": moment_decay,
            "beta1_update_interval": beta1_update_interval,
        }
        if state_dtype is not None:
            kwargs["state_dtype"] = state_dtype
        return STAMLite(**kwargs)

    return _as_optax_transform(make_optimizer)
