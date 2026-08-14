#!/usr/bin/env python3
import argparse

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from style import plt, prepare_output, set_style


def load_data(path):
    required = {"corr_mat_A", "corr_mat_B", "mean_A", "mean_B", "bootstrap_corrs"}
    with np.load(path) as data:
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing NPZ fields: {sorted(missing)}")
        return {name: np.asarray(data[name]) for name in required}


def draw_summary(data, output):
    bootstrap = np.asarray(data["bootstrap_corrs"], dtype=float).ravel()
    means = [float(data["mean_A"]), float(data["mean_B"]), float(np.mean(bootstrap))]
    errors = [0.0, 0.0, float(np.std(bootstrap))]
    labels = ["Synthesized", "NSD-Synthetic\nidx 0-15", "NSD-Synthetic\nbootstrap"]
    colors = ["#4c78a8", "#f28e2b", "#808080"]
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.bar(np.arange(3), means, yerr=errors, color=colors, capsize=5, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(np.arange(3), labels=labels)
    ax.set_ylabel("Mean inter-layer RSM correlation", fontsize=13)
    ax.set_ylim(-0.25, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")


def draw_matrix(matrix, output):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got {matrix.shape}")
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, interpolation="nearest")
    labels = np.arange(1, matrix.shape[0] + 1)
    ax.set_xticks(labels - 1, labels=labels)
    ax.set_yticks(labels - 1, labels=labels)
    ax.set_xlabel("InceptionV1 layer", fontsize=13)
    ax.set_ylabel("InceptionV1 layer", fontsize=13)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.04)
    colorbar.set_label("RSM correlation", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot one NSD-Synthetic comparison panel.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--panel", required=True, choices=("summary", "synthesized", "nsd"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = load_data(args.data)
    set_style()
    output = prepare_output(args.output)
    if args.panel == "summary":
        draw_summary(data, output)
    elif args.panel == "synthesized":
        draw_matrix(data["corr_mat_A"], output)
    else:
        draw_matrix(data["corr_mat_B"], output)


if __name__ == "__main__":
    main()
