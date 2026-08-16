"""Render the synthesized unthresholded t maps with Figure-3 FDR outlines.

The final Figure 3 uses one-sample t statistics across the 30 participants'
permutation-normalized z maps, followed by one-sided Benjamini-Hochberg FDR
within the representational-stability mask. This script displays those same
unthresholded t maps and overlays the same inferential result in black. It
replaces the earlier S14A rendering based on Stouffer group z.
"""
import argparse
import os
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import N_LAYERS, REBUTTAL_OUTPUT, SYNTHESIZED_SUBS  # noqa: E402

sys.path.append("display_surface_plot")
from display_surface_plot import (  # noqa: E402
    fdr_significant_mask,
    plot_surface_saveImageReload_pdf,
    scratchResultFolder,
)

NC_VIZ_DIR = os.path.join(REBUTTAL_OUTPUT, "noise_ceiling_viz")
PAPER_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper_figures")
OUT_PDF = os.path.join(PAPER_FIG_DIR, "paper_figure_outlined.pdf")
STABILITY_MASK_PATH = os.path.join(
    scratchResultFolder, "repStability_RSA_sl", "rad3", "__tfce_corrp_tstat1.nii.gz"
)
FDR_ALPHA = 0.05
VMAX = 8.0  # same t-statistic scale used for final Figure 3


def layer_base(layer):
    return os.path.join(
        scratchResultFolder,
        "findLayerInBrain_leaveOneSubOutReg",
        "shuffleNorm",
        "randomise",
        f"layer{layer}",
        "reg",
        "rad3",
    )


def contour_mask_path(layer):
    return os.path.join(
        NC_VIZ_DIR, f"fdr_sig_mask_layer{layer}_synthesized_tstat.nii.gz"
    )


def _source_paths(layer):
    return (
        os.path.join(layer_base(layer), "__tstat1.nii.gz"),
        os.path.join(layer_base(layer), "subs_compiled.nii.gz"),
    )


def validate_fdr_mask(layer):
    tstat_path, _ = _source_paths(layer)
    mask_path = contour_mask_path(layer)
    for path, label in ((tstat_path, "t-statistic map"), (mask_path, "FDR mask")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Layer {layer}: missing {label}: {path}")
    tstat_nii = nib.load(tstat_path)
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata()
    if mask.shape != tstat_nii.shape:
        raise ValueError(
            f"Layer {layer}: FDR mask shape {mask.shape} != t map {tstat_nii.shape}"
        )
    if not np.allclose(mask_nii.affine, tstat_nii.affine, atol=1e-4):
        raise ValueError(f"Layer {layer}: FDR mask affine does not match t map")
    if not np.all(np.isfinite(mask)) or not np.all(np.isin(np.unique(mask), [0, 1])):
        raise ValueError(f"Layer {layer}: FDR mask is not finite binary 0/1: {mask_path}")


def validate_pdf_output(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected output PDF was not created: {path}")
    if os.path.getsize(path) < 10000:
        raise ValueError(f"Output PDF is unexpectedly small/corrupt: {path}")
    with open(path, "rb") as handle:
        header = handle.read(5)
        handle.seek(max(0, os.path.getsize(path) - 4096))
        trailer = handle.read()
    if header != b"%PDF-" or b"%%EOF" not in trailer:
        raise ValueError(f"Output file failed PDF header/trailer validation: {path}")


def save_fdr_mask(layer):
    tstat_path, subs_path = _source_paths(layer)
    for path, label in (
        (tstat_path, "t-statistic map"),
        (subs_path, "compiled participant maps"),
        (STABILITY_MASK_PATH, "representational-stability mask"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Layer {layer}: missing {label}: {path}")

    tstat_nii = nib.load(tstat_path)
    tstat = tstat_nii.get_fdata()
    if not np.all(np.isfinite(tstat)):
        raise ValueError(f"Layer {layer}: t-statistic map contains non-finite values")
    subs_nii = nib.load(subs_path)
    if subs_nii.ndim != 4:
        raise ValueError(f"Layer {layer}: {subs_path} is not 4D")
    n_subjects = int(subs_nii.shape[3])
    expected = len(SYNTHESIZED_SUBS)
    if n_subjects != expected:
        raise ValueError(
            f"Layer {layer}: compiled N={n_subjects}, expected {expected} participants"
        )
    if subs_nii.shape[:3] != tstat.shape:
        raise ValueError(
            f"Layer {layer}: compiled map shape {subs_nii.shape[:3]} != t map {tstat.shape}"
        )
    if not np.allclose(subs_nii.affine, tstat_nii.affine, atol=1e-4):
        raise ValueError(f"Layer {layer}: compiled-map affine does not match t map")

    stability_nii = nib.load(STABILITY_MASK_PATH)
    stability = stability_nii.get_fdata()
    if stability.shape != tstat.shape:
        raise ValueError(
            f"Layer {layer}: stability mask shape {stability.shape} "
            f"does not match t map {tstat.shape}"
        )
    if not np.allclose(stability_nii.affine, tstat_nii.affine, atol=1e-4):
        raise ValueError(f"Layer {layer}: stability-mask affine does not match t map")
    if not np.all(np.isfinite(stability)):
        raise ValueError(f"Layer {layer}: stability mask contains non-finite values")
    in_mask = (stability > 0.95).astype(float)
    significant, info = fdr_significant_mask(
        tstat, in_mask, df=n_subjects - 1, q=FDR_ALPHA
    )
    os.makedirs(NC_VIZ_DIR, exist_ok=True)
    nib.save(
        nib.Nifti1Image(significant.astype(np.uint8), tstat_nii.affine),
        contour_mask_path(layer),
    )
    validate_fdr_mask(layer)
    print(
        f"Layer {layer:2d}: N={n_subjects}, df={n_subjects - 1}, "
        f"mask voxels={info['m']}, FDR significant={info['n_sig']}"
    )


def save_md_png_preview(pdf_path):
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        print("Skipping optional PNG preview: pdftoppm is not installed.")
        return
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    preview_dir = os.path.join(os.path.dirname(pdf_path), "md_png")
    os.makedirs(preview_dir, exist_ok=True)
    prefix = os.path.join(preview_dir, stem)
    subprocess.run(
        [pdftoppm, "-png", "-r", "150", "-f", "1", "-singlefile", pdf_path, prefix],
        check=True,
    )
    print(f"Saved preview: {prefix}.png")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--layer", type=int, choices=range(1, N_LAYERS + 1), default=None,
        help="Render one layer's PNG cache (Slurm-array stage).",
    )
    group.add_argument(
        "--assemble", action="store_true",
        help="Assemble the final PDF from all layer caches.",
    )
    args = parser.parse_args()

    if args.assemble:
        for layer in range(1, N_LAYERS + 1):
            validate_fdr_mask(layer)
    elif args.layer is not None:
        save_fdr_mask(args.layer)
    else:
        for layer in range(1, N_LAYERS + 1):
            save_fdr_mask(layer)

    common = dict(
        resultType="synthesized: tstat unthresholded (FDR outline)",
        cmap="RdBu_r",
        vmax=VMAX,
        displayThreshold=None,
        bgDarkness=0.3,
        contourMaskPathFn=contour_mask_path,
        views=["posterior"],
        figureName="paper_figure_outlined_tstatFDR_posterior_",
        copyTo=OUT_PDF,
        colorBarVmin=-VMAX,
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
        print(f"Saved PNG cache for layer {args.layer}")
        return
    else:
        plot_surface_saveImageReload_pdf(**common, forceRun=True)

    validate_pdf_output(OUT_PDF)
    save_md_png_preview(OUT_PDF)
    print(f"Saved synthesized unthresholded-t figure: {OUT_PDF}")


if __name__ == "__main__":
    main()
