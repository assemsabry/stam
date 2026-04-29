"""Stress benchmarks for STAM under noisy and non-stationary gradients."""

import argparse
import os
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import random
import optax

from stam_optimizer import STAM, STAMLite
from stam_optimizer.benchmarks.utils import RunResult, block_until_ready_tree, loss_stability, save_results, summarize_results, timing_summary


Params = List[Dict[str, jnp.ndarray]]


class FixedBetaSTAM(STAM):
    """STAM update path with adaptive beta disabled."""

    def __init__(self, **kwargs):
        kwargs["adapt_strength"] = 0.0
        super().__init__(**kwargs)


def init_mlp_params(seed: int, layer_sizes: List[int]) -> Params:
    key = random.PRNGKey(seed)
    keys = random.split(key, len(layer_sizes) - 1)
    params = []
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        params.append({
            "w": random.normal(keys[i], (n_in, n_out)) * jnp.sqrt(2.0 / n_in),
            "b": jnp.zeros((n_out,)),
        })
    return params


def make_shifted_dataset(seed: int, n_samples: int = 2048, n_features: int = 64, n_classes: int = 4):
    key = random.PRNGKey(seed)
    key, w_key, x1_key, x2_key = random.split(key, 4)
    true_w_a = random.normal(w_key, (n_features, n_classes))
    true_w_b = true_w_a + 0.75 * random.normal(key, (n_features, n_classes))

    x_a = random.normal(x1_key, (n_samples // 2, n_features))
    x_b = random.normal(x2_key, (n_samples // 2, n_features)) + 0.7
    y_a = jnp.argmax(x_a @ true_w_a, axis=-1)
    y_b = jnp.argmax(x_b @ true_w_b, axis=-1)

    x = jnp.concatenate([x_a, x_b], axis=0)
    y = jnp.concatenate([y_a, y_b], axis=0)
    return x, y


def mlp_forward(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def loss_fn(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = mlp_forward(params, x)
    labels = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))


def accuracy(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> float:
    preds = jnp.argmax(mlp_forward(params, x), axis=-1)
    return float(jnp.mean(preds == y))


def batch_iter(seed: int, x: jnp.ndarray, y: jnp.ndarray, batch_size: int, steps: int):
    key = random.PRNGKey(seed)
    n = x.shape[0]
    for _ in range(steps):
        key, subkey = random.split(key)
        idx = random.randint(subkey, (batch_size,), minval=0, maxval=n)
        yield x[idx], y[idx]


def build_optimizer(name: str, learning_rate: float, weight_decay: float):
    if name == "adamw":
        return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "stam_fixed":
        return FixedBetaSTAM(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "stam_full":
        return STAM(learning_rate=learning_rate, weight_decay=weight_decay, adapt_strength=0.2)
    if name == "stam_lite":
        return STAMLite(learning_rate=learning_rate, weight_decay=weight_decay, adapt_strength=0.2)
    raise ValueError(f"Unknown optimizer: {name}")


def apply_stam_updates(params: Params, updates: Params) -> Params:
    return jax.tree_util.tree_map(lambda p, u: p - u, params, updates)


def run_one(
    optimizer_name: str,
    seed: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> RunResult:
    x, y = make_shifted_dataset(seed)
    params = init_mlp_params(seed + 1000, [64, 128, 64, 4])
    optimizer = build_optimizer(optimizer_name, learning_rate, weight_decay)
    opt_state = optimizer.init(params)

    losses: List[float] = []
    accuracies: List[float] = []
    times: List[float] = []

    for xb, yb in batch_iter(seed + 2000, x, y, batch_size, steps):
        start = time.time()
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        if optimizer_name == "adamw":
            params = optax.apply_updates(params, updates)
        else:
            params = apply_stam_updates(params, updates)
        block_until_ready_tree((params, loss))
        times.append(time.time() - start)
        losses.append(float(loss))
        accuracies.append(accuracy(params, x, y))

    timing = timing_summary(times)
    return RunResult(
        benchmark=f"stress_shift_batch_{batch_size}",
        optimizer=optimizer_name,
        seed=seed,
        initial_loss=losses[0],
        final_loss=losses[-1],
        final_accuracy=accuracies[-1],
        mean_epoch_time=float(jnp.mean(jnp.array(times))),
        median_step_time=timing["median_step_time"],
        post_warmup_mean_step_time=timing["post_warmup_mean_step_time"],
        loss_stability_std=loss_stability(losses),
        losses=losses,
        accuracies=accuracies,
    )


def run_stress(seeds: List[int], batch_sizes: List[int], steps: int, output_path: str):
    results: List[RunResult] = []
    optimizers = ["adamw", "stam_full", "stam_lite", "stam_fixed"]

    for batch_size in batch_sizes:
        for optimizer_name in optimizers:
            for seed in seeds:
                print(f"Running stress optimizer={optimizer_name} seed={seed} batch={batch_size} steps={steps}")
                result = run_one(
                    optimizer_name=optimizer_name,
                    seed=seed,
                    batch_size=batch_size,
                    steps=steps,
                    learning_rate=1e-3,
                    weight_decay=0.01,
                )
                print(
                    f"  final_loss={result.final_loss:.4f} "
                    f"acc={result.final_accuracy:.4f} "
                    f"stability={result.loss_stability_std:.6f}"
                )
                results.append(result)

    save_results(results, output_path)
    print("\nSummary:")
    for name, stats in summarize_results(results).items():
        print(
            f"{name}: loss={stats['final_loss_mean']:.4f}±{stats['final_loss_std']:.4f}, "
            f"acc={stats['final_accuracy_mean']:.4f}±{stats['final_accuracy_std']:.4f}, "
            f"stability={stats['loss_stability_std_mean']:.6f}, "
            f"median_step={stats['median_step_time']:.6f}s"
        )
    print(f"\nSaved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run STAM stress benchmarks")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-sizes", type=str, default="4,8,32")
    parser.add_argument("--output", type=str, default=os.path.join("results", "stress_shift.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    run_stress(list(range(args.seeds)), batch_sizes, args.steps, args.output)
