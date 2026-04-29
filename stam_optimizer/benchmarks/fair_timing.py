"""Fair warmed timing benchmark for optimizer train steps."""

import argparse
import json
import os
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from stam_optimizer import STAMLite
from stam_optimizer.benchmarks.phase1_runner import (
    build_optimizer,
    init_mlp_params,
    loss_fn,
    make_synthetic_classification,
)
from stam_optimizer.benchmarks.utils import block_until_ready_tree


def _apply_updates(params: Any, updates: Any, optimizer_name: str) -> Any:
    if optimizer_name == "adamw":
        return optax.apply_updates(params, updates)
    return jax.tree_util.tree_map(lambda p, u: p - u, params, updates)


def _build_timing_optimizer(optimizer_name: str, learning_rate: float, weight_decay: float) -> Any:
    if optimizer_name == "stam_lite_f16":
        return STAMLite(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adapt_strength=0.2,
            beta1_update_interval=5,
            state_dtype=jnp.float16,
        )
    if optimizer_name == "stam_lite_lazy10":
        return STAMLite(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adapt_strength=0.2,
            beta1_update_interval=10,
            state_dtype=jnp.float32,
        )
    if optimizer_name == "stam_lite_every1":
        return STAMLite(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adapt_strength=0.2,
            beta1_update_interval=1,
            state_dtype=jnp.float32,
        )
    return build_optimizer(optimizer_name, learning_rate, weight_decay)


def _make_step(optimizer: Any, optimizer_name: str):
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = _apply_updates(params, updates, optimizer_name)
        return params, opt_state, loss

    return jax.jit(step)


def time_optimizer(
    optimizer_name: str,
    seed: int,
    warmup_steps: int,
    timed_steps: int,
    learning_rate: float,
    weight_decay: float,
) -> Dict[str, Any]:
    (x_train, y_train), _ = make_synthetic_classification(seed)
    params = init_mlp_params(seed + 10_000, [64, 128, 64, 4])
    optimizer = _build_timing_optimizer(optimizer_name, learning_rate, weight_decay)
    opt_state = optimizer.init(params)
    step = _make_step(optimizer, optimizer_name)

    compile_start = time.perf_counter()
    params, opt_state, loss = step(params, opt_state, x_train, y_train)
    block_until_ready_tree((params, opt_state, loss))
    compile_seconds = time.perf_counter() - compile_start

    for _ in range(warmup_steps):
        params, opt_state, loss = step(params, opt_state, x_train, y_train)
        block_until_ready_tree((params, opt_state, loss))

    step_seconds: List[float] = []
    losses: List[float] = []
    for _ in range(timed_steps):
        start = time.perf_counter()
        params, opt_state, loss = step(params, opt_state, x_train, y_train)
        block_until_ready_tree((params, opt_state, loss))
        step_seconds.append(time.perf_counter() - start)
        losses.append(float(loss))

    arr = np.array(step_seconds, dtype=np.float64)
    return {
        "optimizer": optimizer_name,
        "seed": seed,
        "compile_seconds": float(compile_seconds),
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "median_step_seconds": float(np.median(arr)),
        "mean_step_seconds": float(arr.mean()),
        "std_step_seconds": float(arr.std()),
        "min_step_seconds": float(arr.min()),
        "max_step_seconds": float(arr.max()),
        "final_timed_loss": losses[-1],
    }


def run_fair_timing(
    optimizers: List[str],
    seeds: List[int],
    warmup_steps: int,
    timed_steps: int,
    output_path: str,
) -> Dict[str, Any]:
    rows = []
    for optimizer_name in optimizers:
        for seed in seeds:
            print(f"Timing optimizer={optimizer_name} seed={seed}")
            rows.append(
                time_optimizer(
                    optimizer_name=optimizer_name,
                    seed=seed,
                    warmup_steps=warmup_steps,
                    timed_steps=timed_steps,
                    learning_rate=1e-3,
                    weight_decay=0.01,
                )
            )

    summary = {}
    for optimizer_name in optimizers:
        group = [r for r in rows if r["optimizer"] == optimizer_name]
        medians = np.array([r["median_step_seconds"] for r in group], dtype=np.float64)
        compiles = np.array([r["compile_seconds"] for r in group], dtype=np.float64)
        summary[optimizer_name] = {
            "runs": len(group),
            "median_step_seconds_mean": float(medians.mean()),
            "median_step_seconds_std": float(medians.std(ddof=1)) if len(group) > 1 else 0.0,
            "compile_seconds_mean": float(compiles.mean()),
        }

    payload = {
        "benchmark": "fair_warmed_jit_timing",
        "methodology": {
            "compile_time_is_separate": True,
            "warmup_excluded_from_timing": True,
            "block_until_ready_after_each_step": True,
            "timed_unit": "full jitted train step: loss, grad, optimizer update, parameter apply",
        },
        "rows": rows,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {output_path}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Run fair warmed optimizer timing benchmark")
    parser.add_argument("--optimizers", type=str, default="adamw,stam_full,stam_lite")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--output", type=str, default=os.path.join("results", "fair_timing.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    optimizer_names = [x.strip() for x in args.optimizers.split(",") if x.strip()]
    seed_values = list(range(args.seeds))
    run_fair_timing(
        optimizers=optimizer_names,
        seeds=seed_values,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        output_path=args.output,
    )
