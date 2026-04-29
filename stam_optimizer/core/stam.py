"""STAM Optimizer: Stable Training with Adaptive Momentum.

Extends AdamW with variance-adaptive momentum for improved stability
in non-stationary gradient regimes.
"""

from typing import Any, NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_map

from .state import STAMState, init_stam_state


class _STAMUpdateResult(NamedTuple):
    update: jnp.ndarray
    mu: jnp.ndarray
    nu: jnp.ndarray
    sigma_sq: jnp.ndarray
    tau: jnp.ndarray
    b1_prod: jnp.ndarray


class STAM:
    """STAM Optimizer.
    
    **S**table **T**raining with **A**daptive **M**omentum
    
    Core innovation: Adaptive beta1 based on gradient variance.
    When gradient direction changes rapidly (high variance), momentum
    is reduced for stability. When stable, momentum stays high for fast
    convergence.
    
    Mathematical formulation:
        r_t = g_t - m_{t-1}                    # gradient residual
        σ²_t = β2·σ²_{t-1} + (1-β2)·mean(r_t²)
        τ = 0.99·τ + 0.01·mean(|r_t|)          # auto-scaling
        z_t = σ²_t / τ²
        β₁(t) = β1_base · (1 - adapt_strength · z_t / (1 + z_t))
        
    All other updates follow standard AdamW.
    
    Args:
        learning_rate: Learning rate (alpha)
        b1_base: Base first moment decay (default: 0.9)
        b2: Second moment decay (default: 0.999)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Decoupled weight decay (default: 0.01)
        adapt_strength: Strength of beta1 adaptation, in [0, 0.5] (default: 0.2)
        tau_decay: Decay for running scale tau (default: 0.99)
        fallback_threshold: Variance threshold for fallback (default: 100.0)
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        b1_base: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        adapt_strength: float = 0.2,
        tau_decay: float = 0.99,
        fallback_threshold: float = 100.0
    ):
        self.learning_rate = learning_rate
        self.b1_base = b1_base
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay
        self.adapt_strength = adapt_strength
        self.tau_decay = tau_decay
        self.fallback_threshold = fallback_threshold
        
        if not 0.0 <= adapt_strength <= 0.5:
            raise ValueError("adapt_strength must be in [0, 0.5]")
        if not 0.0 < b1_base < 1.0:
            raise ValueError("b1_base must be in (0, 1)")
        if not 0.0 < b2 < 1.0:
            raise ValueError("b2 must be in (0, 1)")
        
    def init(self, params: Any) -> STAMState:
        """Initialize optimizer state.
        
        Args:
            params: PyTree of parameters
            
        Returns:
            Initial STAMState
        """
        return init_stam_state(params)
    
    def update(
        self,
        grads: Any,
        state: STAMState,
        params: Optional[Any] = None
    ) -> Tuple[Any, STAMState]:
        """Compute parameter updates.
        
        Args:
            grads: PyTree of gradients
            state: Current STAMState
            params: Current parameters (for weight decay)
            
        Returns:
            Tuple of (updates, new_state)
        """
        return _stam_update(
            grads,
            state,
            params,
            self.learning_rate,
            self.b1_base,
            self.b2,
            self.eps,
            self.weight_decay,
            self.adapt_strength,
            self.tau_decay,
            self.fallback_threshold
        )


def _compute_beta1_adaptive(
    sigma_sq: jnp.ndarray,
    tau: jnp.ndarray,
    b1_base: float,
    adapt_strength: float,
    fallback_threshold: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute adaptive beta1 with safety mechanisms.
    
    Args:
        sigma_sq: Variance of gradient changes
        tau: Running scale
        b1_base: Base beta1
        adapt_strength: Adaptation strength
        fallback_threshold: Threshold for fallback
        
    Returns:
        Tuple of (beta1_effective, should_fallback_flag)
    """
    # Check for NaN/Inf or extreme values (fallback conditions)
    is_finite = jnp.isfinite(sigma_sq)
    is_normal = sigma_sq < fallback_threshold * (tau ** 2 + 1e-8)
    should_fallback = jnp.logical_or(~is_finite, ~is_normal)
    
    # Normalize variance by running scale
    # tau acts as auto-temperature (no manual tuning needed)
    normalized_var = sigma_sq / (tau ** 2 + 1e-8)
    
    normalized_var = jnp.maximum(normalized_var, 0.0)
    s = normalized_var / (1.0 + normalized_var)
    
    # Adaptive beta1: adapts DOWN from base only
    # b1(t) = b1_base * (1 - adapt_strength * s)
    # When s->1 (high variance): b1 -> b1_base * (1 - adapt_strength)
    # When s=0 (low variance): b1 = b1_base
    b1_adaptive = b1_base * (1.0 - adapt_strength * s)
    
    # Clamp to safe range: never exceed base, never go below 0.5
    b1_min = jnp.maximum(0.5, b1_base * (1.0 - adapt_strength))
    b1_adaptive = jnp.clip(b1_adaptive, b1_min, b1_base)
    
    # Fallback: use base beta1 if variance is unstable
    b1_effective = jnp.where(should_fallback, b1_base, b1_adaptive)
    
    return b1_effective, should_fallback




@jit
def _stam_update_single(
    g: jnp.ndarray,
    m: jnp.ndarray,
    v: jnp.ndarray,
    s: jnp.ndarray,
    t: jnp.ndarray,
    b1p: jnp.ndarray,
    p: Optional[jnp.ndarray],
    count: jnp.ndarray,
    learning_rate: float,
    b1_base: float,
    b2: float,
    eps: float,
    weight_decay: float,
    adapt_strength: float,
    tau_decay: float,
    fallback_threshold: float
) -> _STAMUpdateResult:
    """Update a single parameter element (JIT-compiled).
    
    Returns:
        Tuple of (update, new_m, new_v, new_s, new_t, new_b1_prod)
    """
    safe_g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    residual = safe_g - m
    
    # Update tensor-level residual variance proxy
    sigma_sq_new = b2 * s + (1.0 - b2) * jnp.mean(residual ** 2)
    
    # Update running scale (auto-temperature)
    tau_new = tau_decay * t + (1.0 - tau_decay) * jnp.mean(jnp.abs(residual))
    
    # Compute adaptive beta1 with safety
    b1_effective, _ = _compute_beta1_adaptive(
        sigma_sq_new, tau_new, b1_base, adapt_strength, fallback_threshold
    )
    
    # Update first moment with adaptive beta1
    mu_new = b1_effective * m + (1.0 - b1_effective) * safe_g
    
    # Update second moment (standard Adam)
    nu_new = b2 * v + (1.0 - b2) * (safe_g ** 2)
    
    b1_prod_new = b1p * b1_effective
    b1_correction = 1.0 - b1_prod_new
    mu_hat = mu_new / (b1_correction + eps)
    
    # Bias correction for second moment
    b2_correction = 1.0 - (b2 ** (count + 1))
    nu_hat = nu_new / (b2_correction + eps)
    
    # Compute update (AdamW-style)
    update = learning_rate * mu_hat / (jnp.sqrt(nu_hat) + eps)
    if p is not None:
        update = update + learning_rate * weight_decay * p
    
    return _STAMUpdateResult(update, mu_new, nu_new, sigma_sq_new, tau_new, b1_prod_new)


def _stam_update(
    grads: Any,
    state: STAMState,
    params: Optional[Any],
    learning_rate: float,
    b1_base: float,
    b2: float,
    eps: float,
    weight_decay: float,
    adapt_strength: float,
    tau_decay: float,
    fallback_threshold: float
) -> Tuple[Any, STAMState]:
    """Apply STAM update to all parameters.
    
    This is the main update function that operates on PyTrees.
    """
    count = state.count
    
    if params is None and weight_decay != 0.0:
        raise ValueError("params must be provided when weight_decay is non-zero")
    if params is None:
        params = tree_map(jnp.zeros_like, grads)

    results = tree_map(
        lambda g, m, v, s, t, b1p, p: _stam_update_single(
            g, m, v, s, t, b1p, p, count,
            learning_rate, b1_base, b2, eps, weight_decay,
            adapt_strength, tau_decay, fallback_threshold
        ),
        grads,
        state.mu,
        state.nu,
        state.sigma_sq,
        state.tau,
        state.b1_prod,
        params,
    )

    is_update_result = lambda x: isinstance(x, _STAMUpdateResult)
    updates = tree_map(lambda r: r.update, results, is_leaf=is_update_result)
    mu_new = tree_map(lambda r: r.mu, results, is_leaf=is_update_result)
    nu_new = tree_map(lambda r: r.nu, results, is_leaf=is_update_result)
    sigma_sq_new = tree_map(lambda r: r.sigma_sq, results, is_leaf=is_update_result)
    tau_new = tree_map(lambda r: r.tau, results, is_leaf=is_update_result)
    b1_prod_new = tree_map(lambda r: r.b1_prod, results, is_leaf=is_update_result)
    
    new_state = STAMState(
        count=count + 1,
        mu=mu_new,
        nu=nu_new,
        sigma_sq=sigma_sq_new,
        tau=tau_new,
        b1_prod=b1_prod_new
    )
    
    return updates, new_state
