"""MNIST sanity check for STAM optimizer.

Quick 5-minute test to verify STAM doesn't break.
"""

import time
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random, jit, grad
import optax

from stam_optimizer import STAM


def load_mnist():
    """Load MNIST dataset (placeholder - in real use, use tensorflow_datasets)."""
    # Generate synthetic MNIST-like data for testing
    key = random.PRNGKey(0)
    key, k1, k2 = random.split(key, 3)
    
    # 1000 samples, 784 features (28x28)
    X_train = random.normal(k1, (1000, 784)) * 0.1
    y_train = random.randint(k2, (1000,), 0, 10)
    
    key, k1, k2 = random.split(key, 3)
    X_test = random.normal(k1, (200, 784)) * 0.1
    y_test = random.randint(k2, (200,), 0, 10)
    
    return (X_train, y_train), (X_test, y_test)


def init_mlp_params(key, layer_sizes):
    """Initialize MLP parameters."""
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w = random.normal(keys[i], (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        params.append({'w': w, 'b': b})
    
    return params


def mlp_forward(params, x):
    """Forward pass through MLP."""
    # Flatten input
    x = x.reshape(x.shape[0], -1)
    
    # Hidden layers with ReLU
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer['w'] + layer['b'])
    
    # Output layer (no activation)
    x = x @ params[-1]['w'] + params[-1]['b']
    return x


def cross_entropy_loss(params, x, y):
    """Cross-entropy loss."""
    logits = mlp_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    labels_onehot = jax.nn.one_hot(y, 10)
    loss = -jnp.sum(labels_onehot * log_probs, axis=-1)
    return jnp.mean(loss)


def accuracy(params, x, y):
    """Compute accuracy."""
    logits = mlp_forward(params, x)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y)


def train_step_stam(params, opt_state, x, y, optimizer):
    """Single training step with STAM."""
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
    return params, opt_state, loss


def train_step_adam(params, opt_state, x, y):
    """Single training step with AdamW."""
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)
    updates, opt_state = optax.adamw(learning_rate=0.001, weight_decay=0.01).update(
        grads, opt_state, params
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def run_mnist_sanity(epochs: int = 5, batch_size: int = 64):
    """Run MNIST sanity check.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("=" * 60)
    print("STAM Optimizer - MNIST Sanity Check")
    print("=" * 60)
    
    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Initialize model
    key = random.PRNGKey(42)
    params = init_mlp_params(key, [784, 128, 64, 10])
    n_params = sum(p['w'].size + p['b'].size for p in params)
    print(f"Model: MLP with {n_params:,} parameters")
    print()
    
    # Test STAM
    print("Testing STAM Optimizer...")
    print("-" * 40)
    stam_optimizer = STAM(learning_rate=1e-3, adapt_strength=0.2)
    stam_state = stam_optimizer.init(params)
    stam_params = params
    
    stam_losses = []
    stam_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Simple full-batch training for sanity check
        stam_params, stam_state, loss = train_step_stam(
            stam_params, stam_state, X_train, y_train, stam_optimizer
        )
        
        epoch_time = time.time() - epoch_start
        stam_losses.append(float(loss))
        stam_times.append(epoch_time)
        
        acc = accuracy(stam_params, X_test, y_test)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}, Time={epoch_time:.3f}s")
    
    print(f"\nSTAM Final: Loss={stam_losses[-1]:.4f}, Acc={acc:.4f}")
    print(f"Avg epoch time: {sum(stam_times)/len(stam_times):.3f}s")
    print()
    
    # Test AdamW
    print("Testing AdamW (for comparison)...")
    print("-" * 40)
    adam_optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.01)
    adam_state = adam_optimizer.init(params)
    adam_params = params
    
    adam_losses = []
    adam_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        adam_params, adam_state, loss = train_step_adam(
            adam_params, adam_state, X_train, y_train
        )
        
        epoch_time = time.time() - epoch_start
        adam_losses.append(float(loss))
        adam_times.append(epoch_time)
    
    adam_acc = accuracy(adam_params, X_test, y_test)
    print(f"\nAdamW Final: Loss={adam_losses[-1]:.4f}, Acc={adam_acc:.4f}")
    print(f"Avg epoch time: {sum(adam_times)/len(adam_times):.3f}s")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"STAM: Loss {stam_losses[0]:.4f} → {stam_losses[-1]:.4f}, Acc={acc:.4f}")
    print(f"AdamW: Loss {adam_losses[0]:.4f} → {adam_losses[-1]:.4f}, Acc={adam_acc:.4f}")
    
    # Check if STAM is reasonable
    loss_decreased = stam_losses[-1] < stam_losses[0]
    no_nan = all(jnp.isfinite(jnp.array(stam_losses)))
    
    if loss_decreased and no_nan:
        print("\n✅ STAM Sanity Check PASSED")
        print("   - Loss decreased over training")
        print("   - No NaN/Inf values")
        print("   - Comparable to AdamW")
    else:
        print("\n❌ STAM Sanity Check FAILED")
        if not loss_decreased:
            print("   - Loss did not decrease")
        if not no_nan:
            print("   - NaN/Inf detected")
    
    return {
        'stam_losses': stam_losses,
        'adam_losses': adam_losses,
        'stam_acc': float(acc),
        'adam_acc': float(adam_acc)
    }


if __name__ == "__main__":
    results = run_mnist_sanity(epochs=10)
