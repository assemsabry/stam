"""Benchmark utilities for STAM experiments."""

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np


_T_95_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _mean_95ci_half_width(values: np.ndarray) -> float:
    """Return a two-sided Student-t 95% CI half-width for a sample mean."""
    n = len(values)
    if n < 2:
        return 0.0
    critical = _T_95_CRITICAL.get(n - 1, 1.96)
    return float(critical * values.std(ddof=1) / math.sqrt(n))


@dataclass
class RunResult:
    benchmark: str
    optimizer: str
    seed: int
    final_loss: float
    final_accuracy: float
    initial_loss: float
    mean_epoch_time: float
    median_step_time: float
    post_warmup_mean_step_time: float
    loss_stability_std: float
    losses: List[float]
    accuracies: List[float]


def summarize_results(results: List[RunResult]) -> Dict[str, Any]:
    """Aggregate run results by optimizer."""
    grouped: Dict[str, List[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.optimizer, []).append(result)

    summary = {}
    for optimizer, runs in grouped.items():
        final_losses = np.array([r.final_loss for r in runs], dtype=np.float64)
        final_accs = np.array([r.final_accuracy for r in runs], dtype=np.float64)
        times = np.array([r.mean_epoch_time for r in runs], dtype=np.float64)
        median_step_times = np.array([r.median_step_time for r in runs], dtype=np.float64)
        post_warmup_times = np.array([r.post_warmup_mean_step_time for r in runs], dtype=np.float64)
        stability = np.array([r.loss_stability_std for r in runs], dtype=np.float64)
        n = len(runs)
        summary[optimizer] = {
            "runs": n,
            "final_loss_mean": float(final_losses.mean()),
            "final_loss_std": float(final_losses.std(ddof=1)) if n > 1 else 0.0,
            "final_loss_95ci": _mean_95ci_half_width(final_losses),
            "final_accuracy_mean": float(final_accs.mean()),
            "final_accuracy_std": float(final_accs.std(ddof=1)) if n > 1 else 0.0,
            "final_accuracy_95ci": _mean_95ci_half_width(final_accs),
            "mean_epoch_time": float(times.mean()),
            "median_step_time": float(median_step_times.mean()),
            "post_warmup_mean_step_time": float(post_warmup_times.mean()),
            "loss_stability_std_mean": float(stability.mean()),
            "best_loss": float(final_losses.min()),
            "worst_loss": float(final_losses.max()),
            "median_loss": float(np.median(final_losses)),
        }

    return summary


def save_results(results: List[RunResult], output_path: str) -> None:
    """Save run results and summary to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "created_at_unix": time.time(),
        "runs": [asdict(r) for r in results],
        "summary": summarize_results(results),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def loss_stability(losses: List[float], tail_fraction: float = 0.2) -> float:
    """Compute loss standard deviation on the final part of training."""
    if not np.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be finite and in (0, 1]")
    if not losses:
        return 0.0
    tail_len = max(1, int(len(losses) * tail_fraction))
    tail = jnp.array(losses[-tail_len:])
    return float(jnp.std(tail))


def timing_summary(step_times: List[float], warmup_steps: int = 3) -> Dict[str, float]:
    """Summarize timings while excluding first-step/JIT artifacts."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if any(not np.isfinite(value) or value < 0 for value in step_times):
        raise ValueError("step_times must be finite and non-negative")
    if not step_times:
        return {"median_step_time": 0.0, "post_warmup_mean_step_time": 0.0}
    usable = step_times[min(warmup_steps, len(step_times) - 1):]
    arr = np.array(usable, dtype=np.float64)
    return {
        "median_step_time": float(np.median(arr)),
        "post_warmup_mean_step_time": float(arr.mean()),
    }


def block_until_ready_tree(tree: Any) -> Any:
    """Block on all JAX arrays in a PyTree before measuring elapsed time."""
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree
