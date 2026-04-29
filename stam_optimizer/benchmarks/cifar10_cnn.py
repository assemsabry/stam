"""CIFAR-10 CNN benchmark scaffold for STAM.

This benchmark requires tensorflow-datasets and internet/data access on first run:
    pip install tensorflow-datasets tensorflow

Run:
    python stam_optimizer/benchmarks/cifar10_cnn.py --optimizer stam_full --epochs 5
    python stam_optimizer/benchmarks/cifar10_cnn.py --optimizer stam_lite --epochs 5
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import random
import optax

from stam_optimizer import STAM, STAMLite
from stam_optimizer.benchmarks.utils import RunResult, block_until_ready_tree, save_results, timing_summary


Params = Dict[str, jnp.ndarray]


class FixedBetaSTAM(STAM):
    """STAM update path with adaptive beta disabled."""

    def __init__(self, **kwargs):
        kwargs["adapt_strength"] = 0.0
        super().__init__(**kwargs)


def load_cifar10():
    """Load CIFAR-10 via tensorflow_datasets."""
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow-datasets is required for CIFAR-10. Install with: "
            "pip install tensorflow-datasets tensorflow"
        ) from exc

    train = tfds.load("cifar10", split="train", as_supervised=True, batch_size=-1)
    test = tfds.load("cifar10", split="test", as_supervised=True, batch_size=-1)
    x_train, y_train = tfds.as_numpy(train)
    x_test, y_test = tfds.as_numpy(test)
    x_train = jnp.asarray(x_train, dtype=jnp.float32) / 255.0
    x_test = jnp.asarray(x_test, dtype=jnp.float32) / 255.0
    y_train = jnp.asarray(y_train, dtype=jnp.int32)
    y_test = jnp.asarray(y_test, dtype=jnp.int32)
    return (x_train, y_train), (x_test, y_test)


def init_cnn_params(seed: int) -> Params:
    """Initialize a small CNN."""
    key = random.PRNGKey(seed)
    k1, k2, k3, k4 = random.split(key, 4)
    return {
        "conv1_w": random.normal(k1, (3, 3, 3, 32)) * jnp.sqrt(2.0 / (3 * 3 * 3)),
        "conv1_b": jnp.zeros((32,)),
        "conv2_w": random.normal(k2, (3, 3, 32, 64)) * jnp.sqrt(2.0 / (3 * 3 * 32)),
        "conv2_b": jnp.zeros((64,)),
        "fc1_w": random.normal(k3, (8 * 8 * 64, 128)) * jnp.sqrt(2.0 / (8 * 8 * 64)),
        "fc1_b": jnp.zeros((128,)),
        "fc2_w": random.normal(k4, (128, 10)) * jnp.sqrt(2.0 / 128),
        "fc2_b": jnp.zeros((10,)),
    }


def avg_pool_2x2(x: jnp.ndarray) -> jnp.ndarray:
    """2x2 average pooling."""
    n, h, w, c = x.shape
    x = x.reshape(n, h // 2, 2, w // 2, 2, c)
    return x.mean(axis=(2, 4))


def cnn_forward(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass."""
    x = jax.lax.conv_general_dilated(x, params["conv1_w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    x = jax.nn.relu(x + params["conv1_b"])
    x = avg_pool_2x2(x)
    x = jax.lax.conv_general_dilated(x, params["conv2_w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    x = jax.nn.relu(x + params["conv2_b"])
    x = avg_pool_2x2(x)
    x = x.reshape((x.shape[0], -1))
    x = jax.nn.relu(x @ params["fc1_w"] + params["fc1_b"])
    return x @ params["fc2_w"] + params["fc2_b"]


def loss_fn(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = cnn_forward(params, x)
    labels = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))


def accuracy(params: Params, x: jnp.ndarray, y: jnp.ndarray, max_samples: int = 2048) -> float:
    x = x[:max_samples]
    y = y[:max_samples]
    preds = jnp.argmax(cnn_forward(params, x), axis=-1)
    return float(jnp.mean(preds == y))


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


def run_cifar10(optimizer_name: str, seed: int, epochs: int, batch_size: int, output_path: str):
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    params = init_cnn_params(seed)
    optimizer = build_optimizer(optimizer_name, learning_rate=1e-3, weight_decay=0.01)
    opt_state = optimizer.init(params)

    losses: List[float] = []
    accuracies: List[float] = []
    times: List[float] = []
    key = random.PRNGKey(seed + 1000)
    steps_per_epoch = max(1, x_train.shape[0] // batch_size)

    for epoch in range(epochs):
        key, perm_key = random.split(key)
        perm = random.permutation(perm_key, x_train.shape[0])
        x_epoch = x_train[perm]
        y_epoch = y_train[perm]
        epoch_losses = []
        start = time.time()
        for step in range(steps_per_epoch):
            xb = x_epoch[step * batch_size:(step + 1) * batch_size]
            yb = y_epoch[step * batch_size:(step + 1) * batch_size]
            loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            if optimizer_name == "adamw":
                params = optax.apply_updates(params, updates)
            else:
                params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
            block_until_ready_tree((params, loss))
            epoch_losses.append(float(loss))
        times.append(time.time() - start)
        losses.append(float(jnp.mean(jnp.array(epoch_losses))))
        acc = accuracy(params, x_test, y_test)
        accuracies.append(acc)
        print(f"epoch={epoch + 1} loss={losses[-1]:.4f} acc={acc:.4f} time={times[-1]:.2f}s")

    timing = timing_summary(times)
    result = RunResult(
        benchmark="cifar10_small_cnn",
        optimizer=optimizer_name,
        seed=seed,
        initial_loss=losses[0],
        final_loss=losses[-1],
        final_accuracy=accuracies[-1],
        mean_epoch_time=float(jnp.mean(jnp.array(times))),
        median_step_time=timing["median_step_time"],
        post_warmup_mean_step_time=timing["post_warmup_mean_step_time"],
        loss_stability_std=float(jnp.std(jnp.array(losses[-max(1, epochs // 5):]))),
        losses=losses,
        accuracies=accuracies,
    )
    save_results([result], output_path)
    print(f"Saved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CIFAR-10 CNN benchmark")
    parser.add_argument("--optimizer", choices=["adamw", "stam_fixed", "stam_full", "stam_lite"], default="stam_full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=str, default=os.path.join("results", "cifar10_cnn.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cifar10(args.optimizer, args.seed, args.epochs, args.batch_size, args.output)
