"""STAM Optimizer: Stable Training with Adaptive Momentum

A JAX-based optimizer extending AdamW with variance-adaptive momentum
for improved stability in non-stationary gradient regimes.
"""

from .core.stam import STAM
from .core.stam_lite import STAMLite, STAMLiteState
from .core.state import STAMState
from .optax_transform import STAMOptaxState, stam, stam_lite

__version__ = "0.1.0"
__all__ = ["STAM", "STAMState", "STAMLite", "STAMLiteState", "STAMOptaxState", "stam", "stam_lite"]
