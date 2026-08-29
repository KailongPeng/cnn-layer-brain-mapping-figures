#!/usr/bin/env python3
"""Render a diagnostic Sfig5C matrix from one synthesis checkpoint."""

import argparse
import hashlib
import os
import pickle
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from PIL import Image


N_LAYERS = 12
N_IMAGES = 16
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "_preview_transfer"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_paths(run_dir, mag, iteration):
    checkpoint = run_dir / "mag{}phase0".format(mag) / "currIter{}".format(iteration)
    if not checkpoint.is_dir():
        raise FileNotFoundError("checkpoint directory is missing: {}".format(checkpoint))
    images = [checkpoint / "im{}of16.png".format(i) for i in range(1, 17)]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError("checkpoint images are missing:\n" + "\n".join(missing))
    for path in images:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.width != image.height:
                raise RuntimeError("unexpected checkpoint image: {} {} {}".format(
                    path, image.mode, image.size))
    vars_path = checkpoint / "Vars.pkl"
    if not vars_path.is_file() or vars_path.stat().st_size == 0:
        raise FileNotFoundError("checkpoint Vars.pkl is missing or empty: {}".format(vars_path))
    return checkpoint, images, vars_path


def load_matrix(vars_path, mag, iteration):
    try:
        import pickle5 as checkpoint_pickle
    except ImportError:
        checkpoint_pickle = pickle
    with vars_path.open("rb") as handle:
        data = checkpoint_pickle.load(handle)
    if not hasattr(data, "__len__") or len(data) < 4:
        raise RuntimeError("unexpected Vars.pkl structure")
    container = data[2]
    # Each magnification is run by a separate tfLoop() call, which creates a
    # fresh history container. The checkpoint is written before the current
    # iteration is appended, so currIter100 contains 100 entries (0--99).
    expected_steps = iteration
    if len(container) != expected_steps:
        raise RuntimeError(
            "RSM history has {} steps; expected {} for mag{} iter{} checkpoint-save "
            "ordering".format(len(container), expected_steps, mag, iteration)
        )
    if not container:
        raise RuntimeError("RSM history is empty")

    rsms = np.asarray(container[-1], dtype=np.float64)
    if rsms.shape != (N_LAYERS, N_IMAGES, N_IMAGES):
        raise RuntimeError("unexpected RSM shape: {}".format(rsms.shape))
    if not np.all(np.isfinite(rsms)):
        raise RuntimeError("RSMs contain non-finite values")
    if not np.allclose(rsms, rsms.swapaxes(1, 2), atol=1e-5, rtol=0):
        raise RuntimeError("RSMs are not symmetric")

    upper = np.triu_indices(N_IMAGES, k=1)
    fingerprints = rsms[:, upper[0], upper[1]]
    if np.any(np.std(fingerprints, axis=1) < 1e-10):
        raise RuntimeError("at least one layer fingerprint is effectively constant")
    corr = np.corrcoef(fingerprints)
    if corr.shape != (N_LAYERS, N_LAYERS) or not np.all(np.isfinite(corr)):
        raise RuntimeError("derived layer-correlation matrix is invalid")
    if not np.allclose(corr, corr.T, atol=1e-10, rtol=0):
        raise RuntimeError("derived layer-correlation matrix is not symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-10, rtol=0):
        raise RuntimeError("derived layer-correlation diagonal is not one")
    return rsms, fingerprints, corr, len(container)


def metrics(corr):
    upper = corr[np.triu_indices(N_LAYERS, k=1)]
    adjacent = np.diag(corr, k=1)
    return {
        "mean_offdiagonal_r": float(np.mean(upper)),
        "mean_absolute_offdiagonal_r": float(np.mean(np.abs(upper))),
        "rms_offdiagonal_r": float(np.sqrt(np.mean(upper ** 2))),
        "max_absolute_offdiagonal_r": float(np.max(np.abs(upper))),
        "mean_adjacent_r": float(np.mean(adjacent)),
        "mean_absolute_adjacent_r": float(np.mean(np.abs(adjacent))),
    }


def render_matrix(corr, output_pdf, output_png, mag, iteration, rsm_iteration):
    shown = corr.copy()
    np.fill_diagonal(shown, np.nan)
    cmap = plt.get_cmap("seismic").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    image = ax.imshow(
        np.ma.masked_invalid(shown), cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        interpolation="nearest",
    )
    labels = ["L{}".format(i) for i in range(1, N_LAYERS + 1)]
    ax.set_xticks(range(N_LAYERS), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(N_LAYERS), labels=labels)
    ax.set_xlabel("CNN Layer")
    ax.set_ylabel("CNN Layer")
    ax.set_title(
        "Diagnostic Sfig5C preview: mag{} iter{} images\n"
        "latest RSM stored through iter{}".format(mag, iteration, rsm_iteration)
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_label("Correlation coefficient")
    fig.tight_layout()
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def write_summary(path, run_dir, checkpoint, vars_path, mag, iteration,
                  rsm_steps, values):
    lines = [
        "Sfig5C checkpoint diagnostic",
        "",
        "Run: {}".format(run_dir),
        "Checkpoint: {}".format(checkpoint),
        "Vars.pkl SHA256: {}".format(sha256(vars_path)),
        "Image checkpoint: mag{} iter{} (post-update PNGs)".format(mag, iteration),
        "Latest stored RSM: mag{} iter{} (historical save-order lag)".format(
            mag, iteration - 1),
        "RSM history steps: {}".format(rsm_steps),
        "",
    ]
    lines.extend("{}: {:.6f}".format(key, value) for key, value in values.items())
    lines.extend([
        "",
        "DIAGNOSTIC ONLY: this matrix is suitable for an interim convergence check,",
        "but the final panel must be recomputed from the selected final image checkpoint.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mag", type=int, required=True, choices=range(4))
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint, image_paths, vars_path = checkpoint_paths(
        run_dir, args.mag, args.iter)
    rsms, fingerprints, corr, rsm_steps = load_matrix(
        vars_path, args.mag, args.iter)
    values = metrics(corr)

    stem = "{}_sfig5c_diagnostic_mag{}_iter{}".format(
        run_dir.name, args.mag, args.iter)
    output_dir = args.output_dir.resolve()
    outputs = {
        "pdf": output_dir / (stem + ".pdf"),
        "png": output_dir / (stem + ".png"),
        "npz": output_dir / (stem + ".npz"),
        "txt": output_dir / (stem + ".txt"),
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("no-overwrite gate:\n" + "\n".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
            prefix=".sfig5c_diagnostic_", dir=str(output_dir)) as temporary:
        temporary_dir = Path(temporary)
        temporary_outputs = {
            key: temporary_dir / path.name for key, path in outputs.items()
        }
        render_matrix(
            corr, temporary_outputs["pdf"], temporary_outputs["png"],
            args.mag, args.iter, args.iter - 1)
        np.savez_compressed(
            temporary_outputs["npz"], final_rsms=rsms,
            fingerprints=fingerprints, correlation_fingerprints=corr,
            n_layers=np.array(N_LAYERS), n_images=np.array(N_IMAGES),
            mag=np.array(args.mag),
            image_checkpoint_iteration=np.array(args.iter),
            rsm_checkpoint_iteration=np.array(args.iter - 1),
            image_filenames=np.array([path.name for path in image_paths]),
        )
        write_summary(
            temporary_outputs["txt"], run_dir, checkpoint, vars_path,
            args.mag, args.iter, rsm_steps, values)

        if temporary_outputs["pdf"].stat().st_size > 2 * 1024 * 1024:
            raise RuntimeError("preview PDF unexpectedly exceeds 2 MiB")
        if temporary_outputs["npz"].stat().st_size > 2 * 1024 * 1024:
            raise RuntimeError("matrix cache unexpectedly exceeds 2 MiB")
        for key, destination in outputs.items():
            os.replace(temporary_outputs[key], destination)
    print("SFIG5C CHECKPOINT DIAGNOSTIC PASSED")
    for key, value in values.items():
        print("{}={:.6f}".format(key, value))
    for path in outputs.values():
        print("Saved: {}".format(path))


if __name__ == "__main__":
    main()
