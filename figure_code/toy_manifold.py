#!/usr/bin/env python3
import argparse

import numpy as np

from style import plt, prepare_output, set_style


def main():
    parser = argparse.ArgumentParser(description="Plot one toy-manifold panel.")
    parser.add_argument(
        "--data",
        required=True,
        help="NPZ with layer1, layer2, initial_indices, and optimized_indices",
    )
    parser.add_argument(
        "--panel",
        required=True,
        choices=("layer1-initial", "layer2-initial", "layer1-optimized", "layer2-optimized"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.data) as data:
        required = {"layer1", "layer2", "initial_indices", "optimized_indices"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing NPZ fields: {sorted(missing)}")
        points = np.asarray(data[args.panel.split("-")[0]], dtype=float)
        indices = np.asarray(
            data["initial_indices" if args.panel.endswith("initial") else "optimized_indices"],
            dtype=int,
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Manifold points must have shape n x 3, got {points.shape}")
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("Selected point index is outside the manifold")

    set_style()
    fig = plt.figure(figsize=(5.3, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=7, c="#c7c7c7", alpha=0.35)
    selected = points[indices]
    ax.scatter(selected[:, 0], selected[:, 1], selected[:, 2], s=28, c="#d1495b", alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect(np.ptp(points, axis=0))
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(prepare_output(args.output), bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
