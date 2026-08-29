"""Render Figure S9 with the full split-half matrix and its bar summary.

Panel A shows every Half-A × Half-B layer correlation. Panel B summarizes
the informative diagonal (within-layer reliability) and the off-diagonal
mean (between-layer correlation). Both panels are generated from the same
tracked NPZ cache so the relationship between them can be validated exactly.
"""

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np


N_LAYERS = 12
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    REPO_ROOT
    / "rebuttal/analyses/split_half_data/"
    "split_half_subjects_results.npz"
)
OUT_PDF = (
    REPO_ROOT
    / "figures/final/"
    "Sfig8_Split_half_reliability_across_participants.pdf"
)
OUT_PNG = (
    REPO_ROOT
    / "figures/previews/"
    "Sfig8_Split_half_reliability_across_participants.png"
)

WITHIN_COLOR = "#3A9D90"
BETWEEN_COLOR = "#8C5AA3"


def load_results():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Tracked split-half cache is missing: {DATA_PATH}")

    with np.load(DATA_PATH) as data:
        required = {"within_layer", "between_layer", "n_splits"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"Split-half cache is missing fields: {sorted(missing)}")
        within = np.asarray(data["within_layer"], dtype=float)
        between = np.asarray(data["between_layer"], dtype=float)
        n_splits = int(data["n_splits"])

    if within.shape != (n_splits, N_LAYERS):
        raise RuntimeError(
            f"within_layer shape={within.shape}; expected {(n_splits, N_LAYERS)}"
        )
    if between.shape != (n_splits, N_LAYERS, N_LAYERS):
        raise RuntimeError(
            "between_layer shape="
            f"{between.shape}; expected {(n_splits, N_LAYERS, N_LAYERS)}"
        )
    if not np.all(np.isfinite(within)) or not np.all(np.isfinite(between)):
        raise RuntimeError("Split-half cache contains non-finite values")

    diagonal = np.diagonal(between, axis1=1, axis2=2)
    max_diagonal_difference = float(np.max(np.abs(within - diagonal)))
    if max_diagonal_difference > 1e-12:
        raise RuntimeError(
            "Panel relationship gate failed: the matrix diagonal does not equal "
            f"the bar data (max |difference|={max_diagonal_difference:.3g})"
        )

    return within, between, n_splits


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_matrix(ax, mean_matrix):
    image = ax.imshow(
        mean_matrix,
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    labels = [f"L{i}" for i in range(1, N_LAYERS + 1)]
    ax.set_xticks(np.arange(N_LAYERS), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(N_LAYERS), labels=labels)
    ax.tick_params(labelsize=13, length=4)
    ax.set_xlabel("Layer (Half B)", fontsize=16, labelpad=7)
    ax.set_ylabel("Layer (Half A)", fontsize=16, labelpad=7)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.048, pad=0.035)
    colorbar.set_label("Pearson r", fontsize=15, labelpad=8)
    colorbar.ax.tick_params(labelsize=13, width=1.0, length=4)


def draw_bar(ax, within, between):
    layer_means = np.mean(within, axis=0)
    layer_sds = np.std(within, axis=0)
    layers = np.arange(1, N_LAYERS + 1)

    off_diagonal_mask = ~np.eye(N_LAYERS, dtype=bool)
    per_split_between = np.mean(between[:, off_diagonal_mask], axis=1)
    between_mean = float(np.mean(per_split_between))
    between_sd = float(np.std(per_split_between))

    bars = ax.bar(
        layers,
        layer_means,
        yerr=layer_sds,
        capsize=3,
        color=WITHIN_COLOR,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.95,
        label="Within-layer",
    )
    line = ax.axhline(
        between_mean,
        color=BETWEEN_COLOR,
        linewidth=2.2,
        linestyle="--",
        label="Between-layer",
    )
    ax.axhspan(
        between_mean - between_sd,
        between_mean + between_sd,
        color=BETWEEN_COLOR,
        alpha=0.15,
        linewidth=0,
    )
    ax.axhline(0.0, color="#333333", linewidth=1.0, zorder=3)

    ax.set_xlim(0.25, N_LAYERS + 0.75)
    ax.set_ylim(0.0, 0.69)
    ax.set_xticks(layers)
    ax.tick_params(labelsize=13, length=4)
    ax.set_xlabel("CNN layer", fontsize=16, labelpad=7)
    ax.set_ylabel("Pearson r (split-half reliability)", fontsize=16, labelpad=7)
    ax.legend(
        handles=[line, bars],
        labels=["Between-layer", "Within-layer"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.105),
        ncol=2,
        frameon=False,
        fontsize=12,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return between_mean


def validate_output(expected_between_mean):
    if not OUT_PDF.is_file() or OUT_PDF.stat().st_size < 10_000:
        raise RuntimeError(f"Output PDF is missing or unexpectedly small: {OUT_PDF}")
    if not OUT_PNG.is_file() or OUT_PNG.stat().st_size < 20_000:
        raise RuntimeError(f"Output PNG is missing or unexpectedly small: {OUT_PNG}")

    info = subprocess.run(
        ["pdfinfo", str(OUT_PDF)], capture_output=True, text=True, check=True
    ).stdout
    if "Pages:           1" not in info:
        raise RuntimeError("Output PDF must contain exactly one page")

    fonts = subprocess.run(
        ["pdffonts", str(OUT_PDF)], capture_output=True, text=True, check=True
    ).stdout
    if "LiberationSans" not in fonts:
        raise RuntimeError("Font gate failed: Liberation Sans is not embedded")
    if "DejaVu" in fonts:
        raise RuntimeError("Font gate failed: unexpected DejaVu fallback detected")

    if not 0.018 < expected_between_mean < 0.020:
        raise RuntimeError(
            f"Between-layer mean changed unexpectedly: {expected_between_mean:.6f}"
        )


def main():
    within, between, n_splits = load_results()
    configure_style()

    fig = plt.figure(figsize=(12.1, 5.2))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.28),
        left=0.065,
        right=0.985,
        bottom=0.17,
        top=0.91,
        wspace=0.39,
    )
    ax_matrix = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])

    draw_matrix(ax_matrix, np.mean(between, axis=0))
    between_mean = draw_bar(ax_bar, within, between)

    # Figure-level labels cannot be clipped by either axes' bounding box.
    fig.text(0.012, 0.955, "A", fontsize=24, fontweight="normal", va="top")
    fig.text(
        ax_bar.get_position().x0 - 0.055,
        0.955,
        "B",
        fontsize=24,
        fontweight="normal",
        va="top",
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    validate_output(between_mean)
    print(f"Saved combined S9 PDF: {OUT_PDF}")
    print(f"Saved preview PNG:     {OUT_PNG}")
    print(f"n_splits={n_splits}")
    print(
        "within-layer mean range="
        f"{np.mean(within, axis=0).min():.4f}--{np.mean(within, axis=0).max():.4f}"
    )
    print(f"between-layer off-diagonal mean={between_mean:.4f}")


if __name__ == "__main__":
    main()
