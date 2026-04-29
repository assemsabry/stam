"""STAM-Lite: efficient approximation of adaptive momentum."""

from typing import Any, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_map


class STAMLiteState(NamedTuple):
    """State for STAM-Lite optimizer."""
    count: jnp.ndarray
    mu: Any
    grad_mean: Any
    grad_sq_mean: Any
    beta1: Any
    b1_prod: Any


class _STAMLiteUpdateResult(NamedTuple):
    update: jnp.ndarray
    mu: jnp.ndarray
    grad_mean: jnp.ndarray
    grad_sq_mean: jnp.ndarray
    beta1: jnp.ndarray
    b1_prod: jnp.ndarray


def init_stam_lite_state(
    params: Any,
    state_dtype: Any = jnp.float32,
    b1_base: float = 0.9,
) -> STAMLiteState:
    """Initialize STAM-Lite state."""
    dtype = jnp.dtype(state_dtype)
    return STAMLiteState(
        count=jnp.zeros([], dtype=jnp.int32),
        mu=tree_map(lambda x: jnp.zeros_like(x, dtype=dtype), params),
        grad_mean=tree_map(lambda x: jnp.zeros([], dtype=dtype), params),
        grad_sq_mean=tree_map(lambda x: jnp.zeros([], dtype=dtype), params),
        beta1=tree_map(lambda x: jnp.asarray(b1_base, dtype=dtype), params),
        b1_prod=tree_map(lambda x: jnp.ones([], dtype=dtype), params),
    )


class STAMLite:
    """Efficient approximation of STAM.

    STAMLite approximates STAM's residual variance signal with low-cost gradient
    moments:

        Var(g) ≈ EMA(mean(g²)) - EMA(mean(g))²

    It replaces AdamW's per-parameter second moment with a tensor-level RMS
    denominator, removes the tau EMA, and updates beta1 lazily every k steps.
    This makes it a computational approximation of the full method, not a
    random feature-reduced variant.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        b1_base: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        adapt_strength: float = 0.2,
        moment_decay: float = 0.99,
        beta1_update_interval: int = 5,
        state_dtype: Any = jnp.float32,
    ):
        self.learning_rate = learning_rate
        self.b1_base = b1_base
        self.eps = eps
        self.weight_decay = weight_decay
        self.adapt_strength = adapt_strength
        self.moment_decay = moment_decay
        self.beta1_update_interval = beta1_update_interval
        self.state_dtype = jnp.dtype(state_dtype)

        if not 0.0 <= adapt_strength <= 0.5:
            raise ValueError("adapt_strength must be in [0, 0.5]")
        if not 0.0 < b1_base < 1.0:
            raise ValueError("b1_base must be in (0, 1)")
        if not 0.0 < moment_decay < 1.0:
            raise ValueError("moment_decay must be in (0, 1)")
        if beta1_update_interval < 1:
            raise ValueError("beta1_update_interval must be >= 1")

    def init(self, params: Any) -> STAMLiteState:
        """Initialize optimizer state."""
        return init_stam_lite_state(params, self.state_dtype, self.b1_base)

    def update(
        self,
        grads: Any,
        state: STAMLiteState,
        params: Optional[Any] = None,
    ) -> Tuple[Any, STAMLiteState]:
        """Compute parameter updates."""
        return _stam_lite_update(
            grads,
            state,
            params,
            self.learning_rate,
            self.b1_base,
            self.eps,
            self.weight_decay,
            self.adapt_strength,
            self.moment_decay,
            self.beta1_update_interval,
        )


@jit
def _compute_beta1_lite(
    grad_mean: jnp.ndarray,
    grad_sq_mean: jnp.ndarray,
    beta1_prev: jnp.ndarray,
    count: jnp.ndarray,
    b1_base: float,
    eps: float,
    adapt_strength: float,
    moment_decay: float,
    beta1_update_interval: int,
) -> jnp.ndarray:
    """Compute lazy approximate beta1 for STAM-Lite."""
    moment_correction = 1.0 - moment_decay ** (count + 1)
    grad_mean_hat = grad_mean / (moment_correction + eps)
    grad_sq_mean_hat = grad_sq_mean / (moment_correction + eps)
    variance_proxy = jnp.maximum(grad_sq_mean_hat - grad_mean_hat ** 2, 0.0)
    normalized_var = variance_proxy / (grad_sq_mean_hat + eps)
    s = normalized_var / (1.0 + normalized_var)
    b1_candidate = b1_base * (1.0 - adapt_strength * s)
    b1_min = jnp.maximum(0.5, b1_base * (1.0 - adapt_strength))
    b1_candidate = jnp.clip(b1_candidate, b1_min, b1_base)
    b1_candidate = jnp.where(jnp.isfinite(b1_candidate), b1_candidate, b1_base)
    should_update = (count % beta1_update_interval) == 0
    return jnp.where(should_update, b1_candidate, beta1_prev)


@jit
def _stam_lite_update_single(
    g: jnp.ndarray,
    m: jnp.ndarray,
    grad_mean: jnp.ndarray,
    grad_sq_mean: jnp.ndarray,
    beta1_prev: jnp.ndarray,
    b1p: jnp.ndarray,
    p: jnp.ndarray,
    count: jnp.ndarray,
    learning_rate: float,
    b1_base: float,
    eps: float,
    weight_decay: float,
    adapt_strength: float,
    moment_decay: float,
    beta1_update_interval: int,
) -> _STAMLiteUpdateResult:
    """Update one tensor for STAM-Lite."""
    safe_g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    compute_dtype = safe_g.dtype
    m_compute = m.astype(compute_dtype)
    grad_mean_compute = grad_mean.astype(compute_dtype)
    grad_sq_mean_compute = grad_sq_mean.astype(compute_dtype)
    beta1_prev_compute = beta1_prev.astype(compute_dtype)
    b1p_compute = b1p.astype(compute_dtype)

    g_mean = jnp.mean(safe_g)
    g_sq_mean = jnp.mean(safe_g ** 2)
    grad_mean_new = moment_decay * grad_mean_compute + (1.0 - moment_decay) * g_mean
    grad_sq_mean_new = moment_decay * grad_sq_mean_compute + (1.0 - moment_decay) * g_sq_mean

    beta1_new = _compute_beta1_lite(
        grad_mean_new,
        grad_sq_mean_new,
        beta1_prev_compute,
        count,
        b1_base,
        eps,
        adapt_strength,
        moment_decay,
        beta1_update_interval,
    )
    mu_new = beta1_new * m_compute + (1.0 - beta1_new) * safe_g
    b1_prod_new = b1p_compute * beta1_new
    mu_hat = mu_new / (1.0 - b1_prod_new + eps)
    sq_bias_correction = 1.0 - moment_decay ** (count + 1)
    grad_sq_hat = grad_sq_mean_new / (sq_bias_correction + eps)
    rms_denom = jnp.sqrt(jnp.maximum(grad_sq_hat, 0.0)) + eps

    update = learning_rate * (mu_hat / rms_denom) + learning_rate * weight_decay * p

    return _STAMLiteUpdateResult(
        update,
        mu_new.astype(m.dtype),
        grad_mean_new.astype(grad_mean.dtype),
        grad_sq_mean_new.astype(grad_sq_mean.dtype),
        beta1_new.astype(beta1_prev.dtype),
        b1_prod_new.astype(b1p.dtype),
    )


def _stam_lite_update(
    grads: Any,
    state: STAMLiteState,
    params: Optional[Any],
    learning_rate: float,
    b1_base: float,
    eps: float,
    weight_decay: float,
    adapt_strength: float,
    moment_decay: float,
    beta1_update_interval: int,
) -> Tuple[Any, STAMLiteState]:
    """Apply STAM-Lite update to a PyTree."""
    if params is None and weight_decay != 0.0:
        raise ValueError("params must be provided when weight_decay is non-zero")
    if params is None:
        params = tree_map(jnp.zeros_like, grads)

    results = tree_map(
        lambda g, m, gm, gsm, b1, b1p, p: _stam_lite_update_single(
            g,
            m,
            gm,
            gsm,
            b1,
            b1p,
            p,
            state.count,
            learning_rate,
            b1_base,
            eps,
            weight_decay,
            adapt_strength,
            moment_decay,
            beta1_update_interval,
        ),
        grads,
        state.mu,
        state.grad_mean,
        state.grad_sq_mean,
        state.beta1,
        state.b1_prod,
        params,
    )

    is_update_result = lambda x: isinstance(x, _STAMLiteUpdateResult)
    updates = tree_map(lambda r: r.update, results, is_leaf=is_update_result)
    mu_new = tree_map(lambda r: r.mu, results, is_leaf=is_update_result)
    grad_mean_new = tree_map(lambda r: r.grad_mean, results, is_leaf=is_update_result)
    grad_sq_mean_new = tree_map(lambda r: r.grad_sq_mean, results, is_leaf=is_update_result)
    beta1_new = tree_map(lambda r: r.beta1, results, is_leaf=is_update_result)
    b1_prod_new = tree_map(lambda r: r.b1_prod, results, is_leaf=is_update_result)

    new_state = STAMLiteState(
        count=state.count + 1,
        mu=mu_new,
        grad_mean=grad_mean_new,
        grad_sq_mean=grad_sq_mean_new,
        beta1=beta1_new,
        b1_prod=b1_prod_new,
    )
    return updates, new_state
