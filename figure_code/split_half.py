#!/usr/bin/env python3
import argparse

import numpy as np

from style import plt, prepare_output, set_style


def load_results(path):
    with np.load(path) as data:
        required = {"within_layer", "between_layer"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing NPZ fields: {sorted(missing)}")
        within = np.asarray(data["within_layer"], dtype=float)
        between = np.asarray(data["between_layer"], dtype=float)
    if within.ndim != 2 or between.ndim != 3:
        raise ValueError("within_layer must be 2D and between_layer must be 3D")
    if between.shape[:2] != (within.shape[0], within.shape[1]) or between.shape[2] != within.shape[1]:
        raise ValueError(f"Incompatible shapes: {within.shape} and {between.shape}")
    if not np.allclose(within, np.diagonal(between, axis1=1, axis2=2), atol=1e-12):
        raise ValueError("within_layer does not match the matrix diagonal")
    return within, between


def draw_matrix(between, output):
    mean_matrix = np.mean(between, axis=0)
    n_layers = mean_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    image = ax.imshow(mean_matrix, cmap="Reds", vmin=0.0, vmax=1.0, interpolation="nearest")
    labels = [f"L{i}" for i in range(1, n_layers + 1)]
    ax.set_xticks(np.arange(n_layers), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_layers), labels=labels)
    ax.set_xlabel("Layer, half B", fontsize=14)
    ax.set_ylabel("Layer, half A", fontsize=14)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.04)
    colorbar.set_label("Pearson r", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")


def draw_bar(within, between, output):
    n_layers = within.shape[1]
    layers = np.arange(1, n_layers + 1)
    means = np.mean(within, axis=0)
    errors = np.std(within, axis=0)
    off_diagonal = ~np.eye(n_layers, dtype=bool)
    per_split = np.mean(between[:, off_diagonal], axis=1)
    between_mean = float(np.mean(per_split))
    between_sd = float(np.std(per_split))

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(
        layers,
        means,
        yerr=errors,
        capsize=3,
        color="#3a9d90",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(between_mean, color="#8c5aa3", linewidth=2, linestyle="--")
    ax.axhspan(
        between_mean - between_sd,
        between_mean + between_sd,
        color="#8c5aa3",
        alpha=0.15,
        linewidth=0,
    )
    ax.set_xticks(layers)
    ax.set_xlabel("CNN layer", fontsize=14)
    ax.set_ylabel("Pearson r, split-half reliability", fontsize=14)
    ax.set_xlim(0.25, n_layers + 0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot one split-half panel.")
    parser.add_argument("--data", required=True, help="NPZ with within_layer and between_layer")
    parser.add_argument("--panel", required=True, choices=("matrix", "bar"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    within, between = load_results(args.data)
    set_style()
    output = prepare_output(args.output)
    if args.panel == "matrix":
        draw_matrix(between, output)
    else:
        draw_bar(within, between, output)


if __name__ == "__main__":
    main()
