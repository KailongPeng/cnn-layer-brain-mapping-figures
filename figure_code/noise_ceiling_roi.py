#!/usr/bin/env python3
import argparse

import numpy as np

from style import plt, prepare_output, set_style


def main():
    parser = argparse.ArgumentParser(description="Plot the ROI noise-ceiling panel.")
    parser.add_argument(
        "--data",
        required=True,
        help="NPZ with labels, synthesized_mean, synthesized_sem, nsd_mean, nsd_sem",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.data) as data:
        required = {"labels", "synthesized_mean", "synthesized_sem", "nsd_mean", "nsd_sem"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing NPZ fields: {sorted(missing)}")
        labels = np.asarray(data["labels"]).astype(str)
        synthesized = np.asarray(data["synthesized_mean"], dtype=float)
        synthesized_sem = np.asarray(data["synthesized_sem"], dtype=float)
        nsd = np.asarray(data["nsd_mean"], dtype=float)
        nsd_sem = np.asarray(data["nsd_sem"], dtype=float)

    lengths = {len(labels), len(synthesized), len(synthesized_sem), len(nsd), len(nsd_sem)}
    if len(lengths) != 1:
        raise ValueError("ROI arrays have different lengths")
    if not all(np.all(np.isfinite(x)) for x in (synthesized, synthesized_sem, nsd, nsd_sem)):
        raise ValueError("ROI arrays contain nonfinite values")

    set_style()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(
        x - width / 2,
        synthesized,
        width,
        yerr=synthesized_sem,
        capsize=3,
        color="#4c78a8",
        label="Synthesized images",
    )
    ax.bar(
        x + width / 2,
        nsd,
        width,
        yerr=nsd_sem,
        capsize=3,
        color="#808080",
        label="NSD images",
    )
    ax.set_xticks(x, labels=labels, rotation=45, ha="right")
    ax.set_ylabel("Noise ceiling, Pearson r", fontsize=14)
    ax.legend(frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(prepare_output(args.output), bbox_inches="tight")


if __name__ == "__main__":
    main()
