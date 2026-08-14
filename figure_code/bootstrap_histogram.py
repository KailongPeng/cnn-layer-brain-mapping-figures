#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from style import plt, prepare_output, set_style


def load_samples(path, field):
    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source) as data:
            if field not in data.files:
                raise KeyError(f"{source} has no field named {field!r}")
            values = data[field]
    else:
        values = np.load(source)
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap samples must be finite and nonempty")
    return values


def main():
    parser = argparse.ArgumentParser(description="Plot one bootstrap comparison.")
    parser.add_argument("--samples", required=True, help="NPY or NPZ file")
    parser.add_argument("--field", default="bootstrap")
    parser.add_argument("--observed", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, default=45)
    parser.add_argument("--xlabel", default="Mean cross-layer correlation")
    parser.add_argument("--null-label", default="NSD images")
    parser.add_argument("--observed-label", default="Synthesized images")
    args = parser.parse_args()

    samples = load_samples(args.samples, args.field)
    set_style()
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    ax.hist(samples, bins=args.bins, color="#808080", alpha=0.9, label=args.null_label)
    ax.axvline(
        args.observed,
        color="#1f77b4",
        linewidth=2.5,
        label=args.observed_label,
    )
    ax.set_xlabel(args.xlabel, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(prepare_output(args.output), bbox_inches="tight")


if __name__ == "__main__":
    main()
