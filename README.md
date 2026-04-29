# STAM Optimizer

**S**table **T**raining with **A**daptive **M**omentum

A JAX-based optimizer family with variance-adaptive momentum for improved stability in non-stationary gradient regimes.

## Overview

STAM addresses a key limitation in Adam/AdamW: **fixed momentum** regardless of gradient behavior.

### The Problem

In AdamW, `β₁ = 0.9` is constant:
- Early training (high gradient variance) → fixed high momentum causes overshooting
- Near convergence (low variance) → fixed momentum wastes faster convergence potential

### The Solution

STAM adapts `β₁` based on gradient variance:
```
r_t = g_t - m_{t-1}                          # residual from momentum
σ²_t = EMA(mean(r_t²))                       # tensor-level variance proxy
τ = EMA(mean(|r_t|))                         # tensor-level auto-scaling
z_t = σ²_t / (τ² + ε)
s_t = z_t / (1 + z_t)
β₁(t) = β₁_base · (1 - adapt_strength · s_t)
```

**Behavior:**
- High variance → lower β₁ (reduce momentum, be cautious)
- Low variance → higher β₁ (maintain momentum, converge faster)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### STAM

```python
from stam_optimizer import STAM
import jax

# Initialize optimizer
optimizer = STAM(
    learning_rate=1e-3,
    b1_base=0.9,           # base momentum
    b2=0.999,              # second moment decay
    weight_decay=0.01,     # decoupled weight decay
    adapt_strength=0.2     # adaptation strength [0, 0.5]
)

# Initialize state
params = {'w': jax.numpy.zeros((10, 5)), 'b': jax.numpy.zeros(5)}
state = optimizer.init(params)

# Training loop
for grads in training_loop:
    updates, state = optimizer.update(grads, state, params)
    params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
```

### STAMLite

```python
from stam_optimizer import STAMLite
import jax

optimizer = STAMLite(
    learning_rate=1e-3,
    b1_base=0.9,
    weight_decay=0.01,
    adapt_strength=0.2
)

params = {'w': jax.numpy.zeros((10, 5)), 'b': jax.numpy.zeros(5)}
state = optimizer.init(params)

for grads in training_loop:
    updates, state = optimizer.update(grads, state, params)
    params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
```

## Publishable Variants

This project exposes two publishable optimizer variants:

| Variant | Goal | Trade-off |
|---------|------|-----------|
| `STAM` | AdamW-like stability with adaptive β₁ | Same optimizer-state memory class as AdamW |
| `STAMLite` | Efficient approximation of the same adaptive-momentum objective | Lower adaptive fidelity than full STAM |

`STAMLite` is not intended to be "STAM with fewer features." It explores the trade-off between adaptive fidelity and computational efficiency in momentum-based optimization. Benchmark-only ablations such as `stam_fixed` and `stam_const_beta` are not intended as published variants.

## Mathematical Details

### Core Update Rules

```
r_t = g_t - μ_{t-1}                              # gradient residual
σ²_t = β₂·σ²_{t-1} + (1-β₂)·mean(r_t²)          # tensor-level variance proxy
τ = 0.99·τ + 0.01·mean(|r_t|)                    # tensor-level running scale

s_t = z_t / (1 + z_t), z_t = σ²_t / (τ² + ε)     # zero-baseline variance map
β₁(t) = β₁_base · (1 - α · s_t)                  # adaptive momentum
β₁(t) = clamp(β₁(t), β₁_min, β₁_base)            # safety bounds

μ_t = β₁(t)·μ_{t-1} + (1-β₁(t))·g_t            # first moment (adaptive)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²                 # second moment (standard)

μ̂_t = μ_t / (1 - ∏β₁(i))                         # variable-β₁ bias correction
v̂_t = v_t / (1 - β₂^(t+1))                       # bias correction

update = α · μ̂_t / (√v̂_t + ε)                  # parameter update
θ = θ · (1 - α·λ) - update                       # decoupled weight decay
```

### STAMLite Approximation

STAMLite uses a reduced-cost approximation of the variance signal:

```text
E[g]_t = EMA(mean(g_t))
E[g²]_t = EMA(mean(g_t²))
σ²_t ≈ E[g²]_t - E[g]_t²
z_t = σ²_t / (E[g²]_t + ε)
s_t = z_t / (1 + z_t)
β₁(t) = lazy_update(β₁_base · (1 - α · s_t), every k steps)
RMS_t = sqrt(bias_correct(E[g²]_t)) + ε
update = α · μ̂_t / RMS_t
```

This removes the residual computation, removes the `τ` EMA, replaces AdamW's per-parameter second-moment accumulator with tensor-level RMS normalization, and allows delayed β₁ updates.

### Safety Mechanisms

STAM includes multiple safety fallbacks:
1. **NaN detection**: Falls back to base β₁ if variance is NaN/Inf
2. **Variance explosion**: Caps variance at threshold × τ²
3. **Bounds clamping**: β₁ always in [β₁_min, β₁_base]

### Comparison with AdamW

| Feature | AdamW | STAM | STAMLite |
|---------|-------|------|----------|
| Adaptive β₁ | ❌ Fixed | ✅ Variance-based | ✅ Variance-based |
| Variance signal | EMA(g²) | Residual variance | Moment approximation |
| Second moment | EMA(g²) | Same | ❌ Removed |
| Weight decay | Decoupled | Decoupled | Decoupled |
| Fallback safety | ❌ | ✅ | ✅ |
| Optimizer-state memory | Baseline | Approximately baseline | Lower |
| β₁ update | Fixed | Every step | Lazy every k steps |

### Resource Report
```bash
python stam_optimizer/benchmarks/resource_report.py
```

Current STAM optimizer-state memory is approximately the same as AdamW. STAMLite is designed to reduce optimizer-state memory and update compute with an approximation layer:

```text
small_mlp: AdamW state=0.13MB, STAM state=0.13MB, ratio=1.0005x
medium_mlp: AdamW state=192.07MB, STAM state=192.07MB, ratio=1.0000x
wide_linear: AdamW state=512.06MB, STAM state=512.06MB, ratio=1.0000x
```

STAM is not currently designed to use less optimizer-state memory than AdamW. STAMLite is the efficient approximation variant.

### Current Research Red Flags

Current results are preliminary only:

- Synthetic benchmarks are low-complexity and not representative of ImageNet or LLM training.
- Observed improvements are small and confidence intervals overlap.
- No large-scale validation has been completed yet.
- Timing numbers from one-step smoke tests are not speed claims. JAX timing requires equal JIT compilation boundaries, warmup exclusion, async dispatch blocking, and separation of compile time from steady-state runtime.

Before making performance claims, use real benchmarks and dedicated timing methodology.

### Timing Methodology

Use the dedicated fair timing runner for runtime comparisons:

```bash
python stam_optimizer/benchmarks/fair_timing.py --optimizers stam_full,stam_lite,sgd_momentum,rmsprop,adagrad,nadam,lamb --seeds 2 --warmup-steps 5 --timed-steps 20 --output results/fair_timing.json
```

This benchmark reports compilation time separately, excludes warmup steps, JIT-compiles the full train step for every optimizer, and calls `block_until_ready` after each measured step.

Do not compare `update_seconds` from `phase1_smoke_and_memory.json`; that phase is only for state memory and finite-update validation.

Current local fair-timing smoke results are saved in:

```text
results/fair_timing.json
results/fair_timing_lite_variants.json
results/paper_ready_summary.json
```

These local results show the previous STAM timing gap was a measurement artifact. They are still CPU/local smoke results, not final performance claims.

### Interpreting Simple Convex Results

If SGD with momentum wins on a convex or nearly linear task, this is not a failure mode by itself. In well-conditioned problems with stable gradient directions, momentum can act as direct acceleration while adaptive scaling can slow progress. This supports the paper narrative that adaptive optimizers are not universally superior and should be evaluated by regime.

Poor LAMB behavior on small synthetic/local tasks should be treated as a task-regime mismatch rather than a general claim against LAMB.

## Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `learning_rate` | 1e-3 | > 0 | Step size |
| `b1_base` | 0.9 | (0, 1) | Base first moment decay |
| `b2` | 0.999 | (0, 1) | Second moment decay |
| `weight_decay` | 0.01 | ≥ 0 | Decoupled weight decay |
| `adapt_strength` | 0.2 | [0, 0.5] | How much to adapt β₁ |
| `tau_decay` | 0.99 | (0, 1) | Running scale decay |
| `moment_decay` | 0.99 | (0, 1) | STAMLite gradient-moment decay |
| `beta1_update_interval` | 5 | ≥ 1 | STAMLite lazy β₁ update interval |
| `state_dtype` | float32 | dtype | STAMLite state storage dtype |

## When to Use STAM

### STAM helps when:
- Training with curriculum learning (task changes)
- Early training phases with unstable gradients
- Learning rate schedules with warmup/cooldown
- Multi-modal or non-stationary loss landscapes

### STAM is neutral when:
- Late training convergence (stable gradients)
- Well-conditioned optimization problems
- Already using well-tuned SGD

### STAM may not help when:
- Very small batches (< 8) - noisy variance estimation
- Sparse gradients - undefined variance
- Already stable training - no benefit from adaptation overhead

## Benchmarking

### MNIST Sanity Check
```bash
python stam_optimizer/benchmarks/mnist_sanity.py
```

### Phase 1 Multi-Seed Ablation
```bash
python stam_optimizer/benchmarks/phase1_runner.py --seeds 3 --epochs 20
```

This runs:
- STAM-Full
- STAM-Lite
- SGD + Momentum
- RMSProp
- Adagrad
- NAdam
- LAMB

Results are saved to:
```text
results/phase1_synthetic.json
```

### Optimizer State Resource Report
```bash
python stam_optimizer/benchmarks/resource_report.py
```

Results are saved to:
```text
results/resource_report.json
```

### Non-Stationary Stress Test
```bash
python stam_optimizer/benchmarks/stress_runner.py --seeds 3 --steps 50 --batch-sizes 4,8,32
```

This tests STAM under noisy small-batch gradients and distribution shifts.

Results are saved to:
```text
results/stress_shift.json
```

For fair warmed timing tests, use:
```bash
python stam_optimizer/benchmarks/fair_timing.py --optimizers stam_full,stam_lite,sgd_momentum,rmsprop,adagrad,nadam,lamb --seeds 2 --warmup-steps 5 --timed-steps 20
```

Results are saved to:
```text
results/fair_timing.json
```

### CIFAR-10 Small CNN Benchmark
```bash
pip install tensorflow-datasets tensorflow
python stam_optimizer/benchmarks/cifar10_cnn.py --optimizer stam_full --epochs 5
python stam_optimizer/benchmarks/cifar10_cnn.py --optimizer stam_lite --epochs 5
```

This is the first real-data benchmark scaffold. It may download CIFAR-10 on first run.

## Project Structure

```
stam_optimizer/
├── core/
│   ├── stam.py          # Main optimizer
│   ├── stam_lite.py     # Memory-reduced optimizer
│   ├── state.py         # State management
│   └── __init__.py
├── benchmarks/
│   ├── mnist_sanity.py  # Quick sanity check
│   └── __init__.py
└── __init__.py
```

## Citation

If you use STAM in your research, please cite:

```bibtex
@software{stam_optimizer,
  title={STAM: Stable Training with Adaptive Momentum},
  author={Your Name},
  year={2026},
  url={https://github.com/assemsabry/stam}
}
```

## License

MIT License
