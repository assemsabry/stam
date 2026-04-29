"""Benchmark utilities for STAM experiments."""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np


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
        ci_scale = 1.96 / np.sqrt(max(n, 1))

        summary[optimizer] = {
            "runs": n,
            "final_loss_mean": float(final_losses.mean()),
            "final_loss_std": float(final_losses.std(ddof=1)) if n > 1 else 0.0,
            "final_loss_95ci": float(ci_scale * final_losses.std(ddof=1)) if n > 1 else 0.0,
            "final_accuracy_mean": float(final_accs.mean()),
            "final_accuracy_std": float(final_accs.std(ddof=1)) if n > 1 else 0.0,
            "final_accuracy_95ci": float(ci_scale * final_accs.std(ddof=1)) if n > 1 else 0.0,
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "created_at_unix": time.time(),
        "runs": [asdict(r) for r in results],
        "summary": summarize_results(results),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def loss_stability(losses: List[float], tail_fraction: float = 0.2) -> float:
    """Compute loss standard deviation on the final part of training."""
    if not losses:
        return 0.0
    tail_len = max(1, int(len(losses) * tail_fraction))
    tail = jnp.array(losses[-tail_len:])
    return float(jnp.std(tail))


def timing_summary(step_times: List[float], warmup_steps: int = 3) -> Dict[str, float]:
    """Summarize timings while excluding first-step/JIT artifacts."""
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
