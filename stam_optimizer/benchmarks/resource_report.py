"""Resource report for STAM vs AdamW optimizer states."""

import json
import os
from typing import Any

import jax
import jax.numpy as jnp
import optax

from stam_optimizer import STAM, STAMLite


def pytree_nbytes(tree: Any) -> int:
    """Approximate PyTree array memory in bytes."""
    leaves = jax.tree_util.tree_leaves(tree)
    total = 0
    for leaf in leaves:
        if hasattr(leaf, "nbytes"):
            total += int(leaf.nbytes)
    return total


def build_mlp_params(layer_sizes):
    """Build zero params for memory measurement."""
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params.append({"w": jnp.zeros((n_in, n_out), dtype=jnp.float32), "b": jnp.zeros((n_out,), dtype=jnp.float32)})
    return params


def report_for_model(name: str, params: Any):
    """Report parameter and optimizer-state memory."""
    param_bytes = pytree_nbytes(params)

    adamw = optax.adamw(learning_rate=1e-3, weight_decay=0.01)
    adamw_state = adamw.init(params)
    adamw_bytes = pytree_nbytes(adamw_state)

    stam = STAM(learning_rate=1e-3, weight_decay=0.01)
    stam_state = stam.init(params)
    stam_bytes = pytree_nbytes(stam_state)

    stam_lite = STAMLite(learning_rate=1e-3, weight_decay=0.01)
    stam_lite_state = stam_lite.init(params)
    stam_lite_bytes = pytree_nbytes(stam_lite_state)

    return {
        "model": name,
        "param_bytes": param_bytes,
        "adamw_state_bytes": adamw_bytes,
        "stam_state_bytes": stam_bytes,
        "stam_lite_state_bytes": stam_lite_bytes,
        "stam_vs_adamw_state_ratio": stam_bytes / adamw_bytes if adamw_bytes else None,
        "stam_lite_vs_adamw_state_ratio": stam_lite_bytes / adamw_bytes if adamw_bytes else None,
        "param_mb": param_bytes / (1024 ** 2),
        "adamw_state_mb": adamw_bytes / (1024 ** 2),
        "stam_state_mb": stam_bytes / (1024 ** 2),
        "stam_lite_state_mb": stam_lite_bytes / (1024 ** 2),
    }


def main():
    reports = [
        report_for_model("small_mlp", build_mlp_params([64, 128, 64, 4])),
        report_for_model("medium_mlp", build_mlp_params([1024, 4096, 4096, 1024])),
        report_for_model("wide_linear", build_mlp_params([8192, 8192])),
    ]

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "resource_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print("Resource report:")
    for report in reports:
        print(
            f"{report['model']}: params={report['param_mb']:.2f}MB, "
            f"AdamW state={report['adamw_state_mb']:.2f}MB, "
            f"STAM state={report['stam_state_mb']:.2f}MB, "
            f"STAMLite state={report['stam_lite_state_mb']:.2f}MB, "
            f"STAM ratio={report['stam_vs_adamw_state_ratio']:.4f}x, "
            f"STAMLite ratio={report['stam_lite_vs_adamw_state_ratio']:.4f}x"
        )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
