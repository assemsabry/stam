"""STAM optimizer state management."""

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure


class STAMState(NamedTuple):
    """State for STAM optimizer.
    
    Attributes:
        count: Step counter
        mu: First moment (momentum)
        nu: Second moment (adaptive learning rate denominator)
        sigma_sq: Tensor-level variance proxy
        tau: Running mean of gradient change magnitude (auto-scaling)
    """
    count: jnp.ndarray
    mu: Any  # First moment
    nu: Any  # Second moment
    sigma_sq: Any  # Tensor-level variance proxy
    tau: Any  # Running scale for normalization
    b1_prod: Any


def init_stam_state(params: Any) -> STAMState:
    """Initialize STAM optimizer state.
    
    Args:
        params: PyTree of parameters
        
    Returns:
        Initial STAMState
    """
    zeros_like = lambda x: jnp.zeros_like(x)
    
    return STAMState(
        count=jnp.zeros([], dtype=jnp.int32),
        mu=tree_map(zeros_like, params),
        nu=tree_map(zeros_like, params),
        sigma_sq=tree_map(lambda x: jnp.zeros([], dtype=x.dtype), params),
        tau=tree_map(lambda x: jnp.zeros([], dtype=x.dtype), params),
        b1_prod=tree_map(lambda x: jnp.ones([], dtype=x.dtype), params)
    )
