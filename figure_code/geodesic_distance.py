#!/usr/bin/env python3
import argparse

import numpy as np

from style import plt, prepare_output, set_style


def rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean best-layer index in distance bins from V1."
    )
    parser.add_argument(
        "--data", required=True, help="NPZ with distance_mm and best_layer fields"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bin-width", type=float, default=10.0)
    args = parser.parse_args()

    with np.load(args.data) as data:
        if "distance_mm" not in data.files or "best_layer" not in data.files:
            raise KeyError("The NPZ file must contain distance_mm and best_layer")
        distance = np.asarray(data["distance_mm"], dtype=float).ravel()
        best_layer = np.asarray(data["best_layer"], dtype=float).ravel()
        significant = (
            np.asarray(data["significant"], dtype=bool).ravel()
            if "significant" in data.files
            else np.ones(distance.shape, dtype=bool)
        )

    if not (distance.shape == best_layer.shape == significant.shape):
        raise ValueError("Geodesic fields have different shapes")
    keep = significant & np.isfinite(distance) & np.isfinite(best_layer) & (distance >= 0)
    distance = distance[keep]
    best_layer = best_layer[keep]
    if distance.size < 5:
        raise ValueError("Too few valid voxels")

    edges = np.arange(0, distance.max() + args.bin_width, args.bin_width)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.full(len(centers), np.nan)
    sems = np.full(len(centers), np.nan)
    for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        values = best_layer[(distance >= lower) & (distance < upper)]
        if len(values) >= 5:
            means[i] = np.mean(values)
            sems[i] = np.std(values, ddof=1) / np.sqrt(len(values))

    keep_bins = np.isfinite(means)
    rho = float(np.corrcoef(rank(distance), rank(best_layer))[0, 1])
    print(f"Spearman r={rho:.4f}, n={distance.size}")

    set_style()
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    ax.fill_between(
        centers[keep_bins],
        means[keep_bins] - sems[keep_bins],
        means[keep_bins] + sems[keep_bins],
        color="#4682b4",
        alpha=0.25,
        linewidth=0,
    )
    ax.plot(centers[keep_bins], means[keep_bins], "o-", color="#4682b4", linewidth=2.2)
    ax.set_xlabel("Distance from V1 (mm)", fontsize=14)
    ax.set_ylabel("Mean best-layer index", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(prepare_output(args.output), bbox_inches="tight")


if __name__ == "__main__":
    main()
