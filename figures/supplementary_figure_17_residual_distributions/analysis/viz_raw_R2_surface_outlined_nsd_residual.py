"""Aggregate and render residual-model raw cross-validated R² for Figure S18A."""

import argparse
import os
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (  # noqa: E402
    N_LAYERS,
    N_NSD_SETS,
    NSD_SUBS,
    REBUTTAL_OUTPUT,
    get_nsd_residual_raw_R2_path,
)
from sfig18_nifti_validation import atomic_save_nifti, validate_nifti  # noqa: E402

sys.path.append("display_surface_plot")
from display_surface_plot import plot_surface_saveImageReload_pdf  # noqa: E402

from viz_tstat_outlined_nsd_residual import (  # noqa: E402
    contour_mask_path,
    save_fdr_mask_niis,
    validate_fdr_mask,
)
from viz_tstat_outlined_synthesized import validate_pdf_output  # noqa: E402


NC_VIZ_DIR = os.path.join(REBUTTAL_OUTPUT, "noise_ceiling_viz")
PAPER_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "components")
OUT_PDF = os.path.join(PAPER_FIG_DIR, "raw_R2_surface_outlined_nsd_residual.pdf")
STABILITY_MASK_PATH = (
    "/gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/"
    "nsd/result/repStability_RSA_sl/randomise/rad3/allSet/"
    "__tfce_corrp_tstat1.nii.gz"
)
DEFAULT_VMAX = 0.0037  # residual-specific shared scale; see the S18 vmax diagnostic


def mean_R2_path(layer):
    return os.path.join(NC_VIZ_DIR, f"mean_R2_layer{layer}_nsd_residual.nii.gz")


def mean_R2_is_current(layer):
    output = mean_R2_path(layer)
    if not os.path.isfile(output):
        return False
    source_paths = [
        get_nsd_residual_raw_R2_path(subject, layer, which_set)
        for which_set in range(1, N_NSD_SETS + 1)
        for subject in NSD_SUBS
    ]
    if not all(os.path.isfile(path) for path in source_paths):
        return False
    return os.path.getmtime(output) >= max(os.path.getmtime(path) for path in source_paths)


def aggregate_raw_R2_maps(layers=None, vmax=DEFAULT_VMAX):
    if not os.path.exists(STABILITY_MASK_PATH):
        raise FileNotFoundError(f"Missing NSD stability mask: {STABILITY_MASK_PATH}")
    reference_nii = nib.load(STABILITY_MASK_PATH)
    reference_shape = reference_nii.shape
    voxel_sizes = np.abs(np.diag(reference_nii.affine)[:3])
    if not np.allclose(voxel_sizes, 2.0, atol=0.1):
        raise ValueError(f"NSD stability-mask voxel size is not ~2 mm: {voxel_sizes}")

    os.makedirs(NC_VIZ_DIR, exist_ok=True)
    if layers is None:
        layers = range(1, N_LAYERS + 1)
    for layer in layers:
        total = np.zeros(reference_shape, dtype=np.float64)
        n_maps = 0
        for which_set in range(1, N_NSD_SETS + 1):
            for subject in NSD_SUBS:
                path = get_nsd_residual_raw_R2_path(subject, layer, which_set)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Layer {layer}: missing residual raw R² map "
                        f"({subject}, set {which_set}): {path}"
                    )
                raw_nii = nib.load(path)
                data = raw_nii.get_fdata()
                if data.shape != reference_shape:
                    raise ValueError(
                        f"Layer {layer}: {path} has shape {data.shape}, "
                        f"expected {reference_shape}"
                    )
                if not np.allclose(raw_nii.affine, reference_nii.affine, atol=1e-4):
                    raise ValueError(
                        f"Layer {layer}: raw-map affine does not match the "
                        f"MNI 2-mm reference: {path}"
                    )
                if not np.all(np.isfinite(data)):
                    raise ValueError(f"Layer {layer}: non-finite raw R² values in {path}")
                total += data
                n_maps += 1

        expected = len(NSD_SUBS) * N_NSD_SETS
        if n_maps != expected:
            raise ValueError(f"Layer {layer}: aggregated {n_maps}, expected {expected}")
        mean_r2 = total / n_maps
        atomic_save_nifti(
            mean_r2,
            reference_nii.affine,
            mean_R2_path(layer),
            require_nonzero=True,
        )
        values = mean_r2[mean_r2 != 0]
        if values.size == 0:
            raise ValueError(f"Layer {layer}: aggregated residual raw-R² map is empty")
        print(
            f"Layer {layer:2d}: n={n_maps}, median={np.median(values):.4f}, "
            f"positive={100 * np.mean(values > 0):.2f}%, "
            f"|R²|>{vmax:g}={100 * np.mean(np.abs(values) > vmax):.2f}%"
        )


def validate_mean_and_contour(layer):
    mean_path = mean_R2_path(layer)
    if not mean_R2_is_current(layer):
        raise ValueError(
            f"Layer {layer}: residual mean raw-R² map is missing or older than "
            "one of its 128 source maps"
        )
    reference_nii = nib.load(STABILITY_MASK_PATH)
    validate_nifti(
        mean_path,
        expected_shape=reference_nii.shape,
        expected_affine=reference_nii.affine,
        require_nonzero=True,
    )
    validate_fdr_mask(layer)


def save_md_png_preview(pdf_path):
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        print("Skipping optional PNG preview: pdftoppm is not installed.")
        return
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    preview_dir = os.path.join(os.path.dirname(pdf_path), "..", "previews")
    os.makedirs(preview_dir, exist_ok=True)
    preview_path = os.path.join(preview_dir, stem)
    subprocess.run(
        [
            pdftoppm,
            "-png",
            "-r",
            "150",
            "-f",
            "1",
            "-singlefile",
            pdf_path,
            preview_path,
        ],
        check=True,
    )
    print(f"Saved preview: {preview_path}.png")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--layer", type=int, choices=range(1, N_LAYERS + 1), default=None
    )
    group.add_argument("--assemble", action="store_true")
    parser.add_argument("--vmax", type=float, default=DEFAULT_VMAX)
    args = parser.parse_args()
    if args.vmax <= 0:
        parser.error("--vmax must be positive")

    if args.assemble:
        required_layers = list(range(1, N_LAYERS + 1))
    elif args.layer is not None:
        required_layers = [args.layer]
    else:
        required_layers = list(range(1, N_LAYERS + 1))

    missing_means = [layer for layer in required_layers if not mean_R2_is_current(layer)]
    if missing_means:
        if args.assemble:
            raise FileNotFoundError(
                f"Cannot assemble: missing residual mean raw-R² layers {missing_means}"
            )
        aggregate_raw_R2_maps(missing_means, vmax=args.vmax)

    missing_masks = [
        layer for layer in required_layers if not os.path.exists(contour_mask_path(layer))
    ]
    if missing_masks:
        if args.assemble:
            raise FileNotFoundError(
                f"Cannot assemble: missing residual FDR masks {missing_masks}"
            )
        save_fdr_mask_niis(missing_masks)

    for layer in required_layers:
        validate_mean_and_contour(layer)

    common = dict(
        resultType="NSD residual: raw R2 unthresholded (FDR outline)",
        cmap="RdBu_r",
        vmax=args.vmax,
        displayThreshold=None,
        bgDarkness=0.3,
        contourMaskPathFn=contour_mask_path,
        views=["posterior"],
        figureName="raw_R2_surface_outlined_nsd_residual_posterior_",
        copyTo=OUT_PDF,
        colorBarVmin=-args.vmax,
        colorBarLabel="raw cross-validated $R^2$",
        colorBarTickFormat="%.4f",
    )

    if args.assemble:
        plot_surface_saveImageReload_pdf(
            **common, forceRun=False, assembleOnly=True
        )
    elif args.layer is not None:
        plot_surface_saveImageReload_pdf(
            **common, forceRun=True, layers=[args.layer], renderOnly=True
        )
        print(f"Saved residual raw-R² PNG cache for layer {args.layer}")
        return
    else:
        plot_surface_saveImageReload_pdf(**common, forceRun=True)

    validate_pdf_output(OUT_PDF)
    save_md_png_preview(OUT_PDF)
    print(f"Saved residual raw-R² surface figure: {OUT_PDF}")


if __name__ == "__main__":
    main()
