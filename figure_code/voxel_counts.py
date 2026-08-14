#!/usr/bin/env python3
import argparse

import numpy as np

from style import plt, prepare_output, set_style


def main():
    parser = argparse.ArgumentParser(
        description="Plot significant voxel counts for each NSD set and model layer."
    )
    parser.add_argument("--counts", required=True, help="NPY array, set x layer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    counts = np.asarray(np.load(args.counts), dtype=float)
    if counts.ndim != 2 or np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError(f"Expected a finite nonnegative 2D array, got {counts.shape}")

    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(counts, cmap="cool", vmin=0, vmax=args.vmax, aspect="auto")
    ax.set_xticks(np.arange(counts.shape[1]), labels=np.arange(1, counts.shape[1] + 1))
    ax.set_yticks(np.arange(counts.shape[0]), labels=np.arange(1, counts.shape[0] + 1))
    ax.set_xlabel("CNN layer", fontsize=14)
    ax.set_ylabel("NSD image set", fontsize=14)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.04)
    colorbar.set_label("Significant voxels", fontsize=13)
    fig.tight_layout()
    fig.savefig(prepare_output(args.output), bbox_inches="tight")


if __name__ == "__main__":
    main()
