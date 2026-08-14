#!/usr/bin/env python3
import argparse

import numpy as np

from style import plt, prepare_output, set_style


def mean_layer_correlation(features, indices):
    n = len(indices)
    triangle = np.triu_indices(n, k=1)
    vectors = []
    for layer in features:
        subset = layer[np.ix_(indices, indices)]
        vectors.append(subset[triangle])
    correlations = np.corrcoef(np.asarray(vectors))
    return float(np.mean(correlations[np.triu_indices(len(features), k=1)]))


def main():
    parser = argparse.ArgumentParser(
        description="Plot the mean and sampling variation of cross-layer correlation."
    )
    parser.add_argument("--features", required=True, help="NPY array, layer x image x image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--minimum-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = np.asarray(np.load(args.features), dtype=float)
    if features.ndim != 3 or features.shape[1] != features.shape[2]:
        raise ValueError(f"Expected layer x image x image, got {features.shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature matrices contain nonfinite values")

    total_images = features.shape[1]
    if not 3 <= args.minimum_images <= total_images:
        raise ValueError("minimum-images is outside the available image range")

    rng = np.random.default_rng(args.seed)
    image_counts = np.arange(args.minimum_images, total_images + 1)
    means = []
    standard_deviations = []
    for count in image_counts:
        values = []
        for _ in range(args.iterations):
            indices = np.sort(rng.choice(total_images, count, replace=False))
            values.append(mean_layer_correlation(features, indices))
        means.append(np.mean(values))
        standard_deviations.append(np.std(values))

    set_style()
    fig, left = plt.subplots(figsize=(6.1, 4.6))
    line_sd = left.plot(
        image_counts,
        standard_deviations,
        "o-",
        color="#4682b4",
        linewidth=2,
        markersize=5,
        label="SD of inter-layer r",
    )[0]
    left.set_xlabel("Number of images", fontsize=14)
    left.set_ylabel("SD of inter-layer RSM correlation", fontsize=13, color="#4682b4")
    left.tick_params(axis="y", labelcolor="#4682b4")
    left.set_xticks(image_counts)

    right = left.twinx()
    line_mean = right.plot(
        image_counts,
        means,
        "s--",
        color="#2e8b57",
        linewidth=1.8,
        markersize=4,
        label="Mean inter-layer r",
    )[0]
    right.axhline(0, color="#777777", linestyle=":", linewidth=1)
    right.set_ylim(-0.2, 0.2)
    right.set_ylabel("Mean inter-layer RSM correlation", fontsize=13, color="#2e8b57")
    right.tick_params(axis="y", labelcolor="#2e8b57")
    left.legend(handles=[line_sd, line_mean], frameon=False, fontsize=10, loc="upper right")
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(prepare_output(args.output), bbox_inches="tight")


if __name__ == "__main__":
    main()
