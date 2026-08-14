#!/usr/bin/env python3
import argparse

import numpy as np

from style import plt, prepare_output, set_style


def main():
    parser = argparse.ArgumentParser(
        description="Plot representational stability against within-image similarity."
    )
    parser.add_argument(
        "--data", required=True, help="NPZ with within, between, and significant fields"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-points", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with np.load(args.data) as data:
        required = {"within", "between", "significant"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing NPZ fields: {sorted(missing)}")
        within = np.asarray(data["within"], dtype=float).ravel()
        between = np.asarray(data["between"], dtype=float).ravel()
        significant = np.asarray(data["significant"], dtype=bool).ravel()

    if not (within.shape == between.shape == significant.shape):
        raise ValueError("Scatter fields have different shapes")
    valid = np.isfinite(within) & np.isfinite(between)
    within = within[valid]
    between = between[valid]
    significant = significant[valid]

    if within.size > args.maximum_points:
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(within.size, args.maximum_points, replace=False)
        within = within[selected]
        between = between[selected]
        significant = significant[selected]

    stability = within - between
    set_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.scatter(
        within[~significant],
        stability[~significant],
        s=2,
        c="#9a9a9a",
        alpha=0.18,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        within[significant],
        stability[significant],
        s=3,
        c="#d1495b",
        alpha=0.3,
        linewidths=0,
        rasterized=True,
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Within-image similarity", fontsize=14)
    ax.set_ylabel("Within minus between similarity", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(prepare_output(args.output), bbox_inches="tight")


if __name__ == "__main__":
    main()
