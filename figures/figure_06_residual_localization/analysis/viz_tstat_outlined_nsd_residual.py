"""Render residual-model unthresholded t maps with Figure-6 FDR outlines."""

import argparse
import os
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import N_LAYERS, REBUTTAL_OUTPUT  # noqa: E402
from sfig18_nifti_validation import atomic_save_nifti  # noqa: E402

sys.path.append("figures/figure_06_residual_localization/analysis/display_surface_plot")
from display_surface_plot import (  # noqa: E402
    _nsd_residual_meanacross_tstat_and_mask,
    fdr_significant_mask,
    plot_surface_saveImageReload_pdf,
    scratchResultFolder,
)

from viz_tstat_outlined_synthesized import validate_pdf_output  # noqa: E402


NC_VIZ_DIR = os.path.join(REBUTTAL_OUTPUT, "noise_ceiling_viz")
PAPER_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "components")
OUT_PDF = os.path.join(PAPER_FIG_DIR, "tstat_surface_outlined_nsd_residual.pdf")
FDR_ALPHA = 0.05
DEFAULT_VMAX = 11.0


def contour_mask_path(layer):
    return os.path.join(
        NC_VIZ_DIR, f"fdr_sig_mask_layer{layer}_nsd_residual.nii.gz"
    )


def save_fdr_mask(layer):
    tstat, n_subjects, affine, in_mask = _nsd_residual_meanacross_tstat_and_mask(
        scratchResultFolder, layer
    )
    if not np.all(np.isfinite(tstat)):
        raise ValueError(f"Layer {layer}: reconstructed residual t map is non-finite")
    significant, info = fdr_significant_mask(
        tstat, in_mask, df=n_subjects - 1, q=FDR_ALPHA
    )
    os.makedirs(NC_VIZ_DIR, exist_ok=True)
    atomic_save_nifti(
        significant.astype(np.uint8),
        affine,
        contour_mask_path(layer),
        require_binary=True,
    )
    validate_fdr_mask(layer, expected_affine=affine, expected_shape=tstat.shape)
    values = tstat[in_mask.astype(bool)]
    print(
        f"Layer {layer:2d}: N={n_subjects}, df={n_subjects - 1}, "
        f"mask voxels={info['m']}, FDR significant={info['n_sig']}, "
        f"t p99={np.percentile(values, 99):.3f}, "
        f"range=[{np.min(values):.3f}, {np.max(values):.3f}]"
    )


def save_fdr_mask_niis(layers=None):
    if layers is None:
        layers = range(1, N_LAYERS + 1)
    for layer in layers:
        save_fdr_mask(layer)


def validate_fdr_mask(layer, expected_affine=None, expected_shape=None):
    path = contour_mask_path(layer)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Layer {layer}: missing residual FDR mask: {path}")
    nii = nib.load(path)
    data = nii.get_fdata()
    if expected_shape is not None and nii.shape != expected_shape:
        raise ValueError(
            f"Layer {layer}: FDR-mask shape {nii.shape} != expected {expected_shape}"
        )
    if expected_affine is not None and not np.allclose(
        nii.affine, expected_affine, atol=1e-4
    ):
        raise ValueError(f"Layer {layer}: FDR-mask affine does not match t map")
    if not np.all(np.isfinite(data)) or not np.all(
        np.isin(np.unique(data), [0, 1])
    ):
        raise ValueError(f"Layer {layer}: FDR mask is not finite binary 0/1")


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
        for layer in range(1, N_LAYERS + 1):
            validate_fdr_mask(layer)
    elif args.layer is not None:
        save_fdr_mask(args.layer)
    else:
        save_fdr_mask_niis()

    common = dict(
        resultType="NSD residual mean+randomise: tstat unthresholded (FDR outline)",
        cmap="RdBu_r",
        vmax=args.vmax,
        displayThreshold=None,
        bgDarkness=0.3,
        contourMaskPathFn=contour_mask_path,
        views=["posterior"],
        figureName="tstat_surface_outlined_nsd_residual_posterior_",
        copyTo=OUT_PDF,
        colorBarVmin=-args.vmax,
        colorBarLabel="t statistic value",
    )

    if args.assemble:
        plot_surface_saveImageReload_pdf(
            **common, forceRun=False, assembleOnly=True
        )
    elif args.layer is not None:
        plot_surface_saveImageReload_pdf(
            **common, forceRun=True, layers=[args.layer], renderOnly=True
        )
        print(f"Saved residual t-map PNG cache for layer {args.layer}")
        return
    else:
        plot_surface_saveImageReload_pdf(**common, forceRun=True)

    validate_pdf_output(OUT_PDF)
    save_md_png_preview(OUT_PDF)
    print(f"Saved residual unthresholded-t figure: {OUT_PDF}")


if __name__ == "__main__":
    main()
