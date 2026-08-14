#!/usr/bin/env python3
import argparse

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from style import plt, prepare_output, set_style


def main():
    parser = argparse.ArgumentParser(description="Plot one cross-layer matrix.")
    parser.add_argument("--matrix", required=True, help="Square NPY matrix")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--xlabel", default="CNN layer")
    parser.add_argument("--ylabel", default="CNN layer")
    parser.add_argument("--colorbar-label", default="Pearson r")
    parser.add_argument("--hide-diagonal", action="store_true")
    args = parser.parse_args()

    matrix = np.asarray(np.load(args.matrix), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got {matrix.shape}")
    if args.vmin >= 0 or args.vmax <= 0:
        raise ValueError("The color range must include zero")
    if args.hide_diagonal:
        matrix = matrix.copy()
        np.fill_diagonal(matrix, np.nan)

    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    norm = TwoSlopeNorm(vmin=args.vmin, vcenter=0.0, vmax=args.vmax)
    image = ax.imshow(matrix, cmap=args.cmap, norm=norm, interpolation="nearest")
    labels = np.arange(1, matrix.shape[0] + 1)
    ax.set_xticks(labels - 1, labels=labels)
    ax.set_yticks(labels - 1, labels=labels)
    ax.set_xlabel(args.xlabel, fontsize=15)
    ax.set_ylabel(args.ylabel, fontsize=15)
    ax.tick_params(labelsize=12)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.04)
    colorbar.set_label(args.colorbar_label, fontsize=14)
    colorbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(prepare_output(args.output), bbox_inches="tight")


if __name__ == "__main__":
    main()
