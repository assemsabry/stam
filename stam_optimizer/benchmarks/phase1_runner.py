"""Phase 1 benchmark runner for STAM.

Runs preliminary multi-seed validation and ablations on synthetic tasks.
This is intentionally lightweight and local-friendly; real CIFAR/LLM benchmarks
come after this infrastructure is validated.
"""

import argparse
import os
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import random
import optax

from stam_optimizer import STAM, STAMLite
from stam_optimizer.benchmarks.utils import RunResult, block_until_ready_tree, loss_stability, save_results, summarize_results, timing_summary


Array = jnp.ndarray
Params = List[Dict[str, Array]]


class FixedBetaSTAM(STAM):
    """STAM state/update path with beta adaptation disabled."""

    def __init__(self, **kwargs):
        kwargs["adapt_strength"] = 0.0
        super().__init__(**kwargs)


class ConstBetaSTAM(STAM):
    """STAM with a lower fixed beta1 to test whether dynamics matter."""

    def __init__(self, **kwargs):
        kwargs["b1_base"] = kwargs.pop("const_b1", 0.81)
        kwargs["adapt_strength"] = 0.0
        super().__init__(**kwargs)


def make_synthetic_classification(seed: int, n_train: int = 1024, n_test: int = 256, n_features: int = 64, n_classes: int = 4):
    """Create a non-stationary synthetic classification task."""
    key = random.PRNGKey(seed)
    key, w_key, x1_key, x2_key, xt_key = random.split(key, 5)
    true_w = random.normal(w_key, (n_features, n_classes))

    x_a = random.normal(x1_key, (n_train // 2, n_features))
    x_b = random.normal(x2_key, (n_train // 2, n_features)) + 0.8
    x_train = jnp.concatenate([x_a, x_b], axis=0)

    logits = x_train @ true_w
    y_train = jnp.argmax(logits, axis=-1)

    x_test = random.normal(xt_key, (n_test, n_features)) + 0.4
    y_test = jnp.argmax(x_test @ true_w, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def init_mlp_params(seed: int, layer_sizes: List[int]) -> Params:
    """Initialize MLP parameters."""
    key = random.PRNGKey(seed)
    keys = random.split(key, len(layer_sizes) - 1)
    params = []
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w = random.normal(keys[i], (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        params.append({"w": w, "b": b})
    return params


def mlp_forward(params: Params, x: Array) -> Array:
    """Forward pass."""
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def loss_fn(params: Params, x: Array, y: Array) -> Array:
    """Cross entropy loss."""
    logits = mlp_forward(params, x)
    labels = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))


def accuracy(params: Params, x: Array, y: Array) -> float:
    """Accuracy."""
    preds = jnp.argmax(mlp_forward(params, x), axis=-1)
    return float(jnp.mean(preds == y))


def apply_negative_updates(params: Params, updates: Params) -> Params:
    """Apply STAM-style positive descent updates."""
    return jax.tree_util.tree_map(lambda p, u: p - u, params, updates)


def build_optimizer(name: str, learning_rate: float, weight_decay: float):
    """Build optimizer by ablation name."""
    if name == "adamw":
        return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return optax.adam(learning_rate=learning_rate)
    if name == "sgd_momentum":
        return optax.sgd(learning_rate=learning_rate, momentum=0.9)
    if name == "rmsprop":
        return optax.rmsprop(learning_rate=learning_rate)
    if name == "adagrad":
        return optax.adagrad(learning_rate=learning_rate)
    if name == "nadam" and hasattr(optax, "nadam"):
        return optax.nadam(learning_rate=learning_rate)
    if name == "lamb" and hasattr(optax, "lamb"):
        return optax.lamb(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "stam_full":
        return STAM(learning_rate=learning_rate, weight_decay=weight_decay, adapt_strength=0.2)
    if name == "stam_lite":
        return STAMLite(learning_rate=learning_rate, weight_decay=weight_decay, adapt_strength=0.2)
    if name == "stam_fixed":
        return FixedBetaSTAM(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "stam_const_beta":
        return ConstBetaSTAM(learning_rate=learning_rate, weight_decay=weight_decay, const_b1=0.81)
    raise ValueError(f"Unknown optimizer: {name}")


def train_one_run(
    optimizer_name: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> RunResult:
    """Train one seed/config."""
    (x_train, y_train), (x_test, y_test) = make_synthetic_classification(seed)
    params = init_mlp_params(seed + 10_000, [64, 128, 64, 4])
    optimizer = build_optimizer(optimizer_name, learning_rate, weight_decay)
    opt_state = optimizer.init(params)

    losses: List[float] = []
    accuracies: List[float] = []
    epoch_times: List[float] = []

    for _ in range(epochs):
        start = time.time()
        loss, grads = jax.value_and_grad(loss_fn)(params, x_train, y_train)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        if optimizer_name in {"adamw", "adam", "sgd_momentum", "rmsprop", "adagrad", "nadam", "lamb"}:
            params = optax.apply_updates(params, updates)
        else:
            params = apply_negative_updates(params, updates)
        block_until_ready_tree((params, loss))
        epoch_times.append(time.time() - start)
        losses.append(float(loss))
        accuracies.append(accuracy(params, x_test, y_test))

    timing = timing_summary(epoch_times)
    return RunResult(
        benchmark="synthetic_nonstationary_mlp",
        optimizer=optimizer_name,
        seed=seed,
        initial_loss=losses[0],
        final_loss=losses[-1],
        final_accuracy=accuracies[-1],
        mean_epoch_time=float(jnp.mean(jnp.array(epoch_times))),
        median_step_time=timing["median_step_time"],
        post_warmup_mean_step_time=timing["post_warmup_mean_step_time"],
        loss_stability_std=loss_stability(losses),
        losses=losses,
        accuracies=accuracies,
    )


def run_phase1(seeds: List[int], epochs: int, output_path: str) -> List[RunResult]:
    """Run Phase 1 ablations."""
    optimizers = ["stam_full", "stam_lite", "sgd_momentum", "rmsprop", "adagrad", "nadam", "lamb"]
    results = []
    for optimizer_name in optimizers:
        for seed in seeds:
            print(f"Running {optimizer_name} seed={seed} epochs={epochs}")
            result = train_one_run(
                optimizer_name=optimizer_name,
                seed=seed,
                epochs=epochs,
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
    for optimizer_name, stats in summarize_results(results).items():
        print(
            f"{optimizer_name}: loss={stats['final_loss_mean']:.4f}±{stats['final_loss_std']:.4f}, "
            f"acc={stats['final_accuracy_mean']:.4f}±{stats['final_accuracy_std']:.4f}, "
            f"median_step={stats['median_step_time']:.6f}s"
        )
    print(f"\nSaved results to {output_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run STAM Phase 1 benchmarks")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=str, default=os.path.join("results", "phase1_synthetic.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_phase1(seeds=list(range(args.seeds)), epochs=args.epochs, output_path=args.output)
