"""Render NSD raw cross-validated R² maps in the same format as synthesized S16.

The original NSD searchlight pipeline saved only permutation-normalized z-score
maps. Run submit_nsd_raw_R2_all_sets.sh first; it invokes the same trained LOSO
model but returns the true R² immediately, without the 500 irrelevant shuffles.

This script then averages the complete 8-subject × 16-set raw maps per layer,
uses the same posterior view and fixed [-3, 3] color scale as synthesized S16,
and overlays the FDR contours already used for the NSD unthresholded t maps.
"""
import os
import sys
import subprocess
import shutil
import argparse

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (  # noqa: E402
    REBUTTAL_OUTPUT,
    NSD_SUBS,
    N_LAYERS,
    N_NSD_SETS,
    get_nsd_raw_R2_path,
)

sys.path.append("display_surface_plot")
from display_surface_plot import plot_surface_saveImageReload_pdf  # noqa: E402
from viz_group_z_outlined_nsd import (  # noqa: E402
    contour_mask_path,
    save_fdr_mask_niis,
)
from viz_tstat_outlined_synthesized import validate_pdf_output  # noqa: E402

NC_VIZ_DIR = os.path.join(REBUTTAL_OUTPUT, "noise_ceiling_viz")
PAPER_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "components")
STABILITY_MASK_PATH = (
    "/gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/"
    "nsd/result/repStability_RSA_sl/randomise/rad3/allSet/"
    "__tfce_corrp_tstat1.nii.gz"
)
OUT_PDF = os.path.join(PAPER_FIG_DIR, "raw_R2_surface_outlined_nsd.pdf")
VMAX = 3.0  # shared with synthesized S16 so the two panels are directly comparable


def mean_R2_path(layer):
    return os.path.join(NC_VIZ_DIR, f"mean_R2_layer{layer}_nsd_nonresidual.nii.gz")


def validate_mean_and_contour(layer):
    mean_path = mean_R2_path(layer)
    mask_path = contour_mask_path(layer)
    for path, label in ((mean_path, "mean raw-R² map"), (mask_path, "FDR mask")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Layer {layer}: missing NSD {label}: {path}")
    reference_nii = nib.load(STABILITY_MASK_PATH)
    mean_nii = nib.load(mean_path)
    mask_nii = nib.load(mask_path)
    for nii, label in ((mean_nii, "mean raw-R² map"), (mask_nii, "FDR mask")):
        if nii.shape != reference_nii.shape:
            raise ValueError(
                f"Layer {layer}: NSD {label} shape {nii.shape} != reference {reference_nii.shape}"
            )
        if not np.allclose(nii.affine, reference_nii.affine, atol=1e-4):
            raise ValueError(f"Layer {layer}: NSD {label} affine != reference affine")
    mean_data = mean_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    if not np.all(np.isfinite(mean_data)):
        raise ValueError(f"Layer {layer}: NSD mean raw-R² map contains non-finite values")
    values = mean_data[mean_data != 0]
    if values.size == 0:
        raise ValueError(f"Layer {layer}: NSD mean raw-R² map is empty")
    n_positive = int(np.sum(values > 0))
    if n_positive:
        raise ValueError(
            f"Layer {layer}: expected uniformly non-positive NSD group raw R², "
            f"found {n_positive} positive voxels"
        )
    if not np.all(np.isfinite(mask_data)) or not np.all(
        np.isin(np.unique(mask_data), [0, 1])
    ):
        raise ValueError(f"Layer {layer}: NSD FDR mask is not finite binary 0/1")


def aggregate_raw_R2_maps(layers=None):
    if not os.path.exists(STABILITY_MASK_PATH):
        raise FileNotFoundError(f"Missing NSD stability mask: {STABILITY_MASK_PATH}")
    mask_nii = nib.load(STABILITY_MASK_PATH)
    ref_shape = mask_nii.shape
    mask_vox = np.abs(np.diag(mask_nii.affine)[:3])
    if not np.allclose(mask_vox, 2.0, atol=0.1):
        raise ValueError(f"NSD stability-mask voxel size is not ~2 mm: {mask_vox}")

    os.makedirs(NC_VIZ_DIR, exist_ok=True)
    if layers is None:
        layers = range(1, N_LAYERS + 1)
    for layer in layers:
        total = np.zeros(ref_shape, dtype=np.float64)
        n_maps = 0
        for which_set in range(1, N_NSD_SETS + 1):
            for subject in NSD_SUBS:
                path = get_nsd_raw_R2_path(subject, layer, which_set)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Layer {layer}: missing NSD raw R² map ({subject}, set {which_set}): {path}"
                    )
                data = nib.load(path).get_fdata()
                if data.shape != ref_shape:
                    raise ValueError(
                        f"Layer {layer}: {path} has shape {data.shape}, expected {ref_shape}"
                    )
                if not np.all(np.isfinite(data)):
                    raise ValueError(f"Layer {layer}: non-finite raw R² values in {path}")
                total += data
                n_maps += 1

        expected = len(NSD_SUBS) * N_NSD_SETS
        if n_maps != expected:
            raise ValueError(f"Layer {layer}: aggregated {n_maps} maps, expected {expected}")
        mean_r2 = total / n_maps
        nib.save(
            nib.Nifti1Image(mean_r2.astype("double"), mask_nii.affine),
            mean_R2_path(layer),
        )
        vals = mean_r2[mean_r2 != 0]
        if vals.size == 0:
            raise ValueError(f"Layer {layer}: aggregated NSD raw R² map is empty")
        print(
            f"  Layer {layer:2d}: n={n_maps}, median={np.median(vals):.4f}, "
            f"positive={100 * np.mean(vals > 0):.2f}%, "
            f"|R²|>{VMAX:g}={100 * np.mean(np.abs(vals) > VMAX):.2f}%"
        )


def save_md_png_preview(pdf_path):
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        print("  Skipping optional PNG preview: pdftoppm is not installed.")
        return

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    md_png_dir = os.path.join(os.path.dirname(pdf_path), "..", "previews")
    os.makedirs(md_png_dir, exist_ok=True)
    prefix = os.path.join(md_png_dir, stem)
    subprocess.run([pdftoppm, "-png", "-r", "150", pdf_path, prefix], check=True)
    rendered = f"{prefix}-1.png"
    final = f"{prefix}.png"
    if not os.path.exists(rendered):
        raise FileNotFoundError(f"pdftoppm did not create {rendered}")
    shutil.move(rendered, final)
    print(f"  Saved preview: {final}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--layer", type=int, choices=range(1, N_LAYERS + 1), default=None,
        help="Render only one layer's PNG cache (Slurm-array stage).",
    )
    group.add_argument(
        "--assemble", action="store_true",
        help="Assemble the final PDF from existing layer caches.",
    )
    args = parser.parse_args()

    print("=== Aggregate and render NSD raw cross-validated R² ===")
    if args.assemble:
        required_layers = range(1, N_LAYERS + 1)
    elif args.layer is not None:
        required_layers = [args.layer]
    else:
        required_layers = range(1, N_LAYERS + 1)

    missing_means = [layer for layer in required_layers if not os.path.exists(mean_R2_path(layer))]
    if missing_means:
        if args.assemble:
            raise FileNotFoundError(
                f"Cannot assemble: missing aggregated raw-R² maps for layers {missing_means}"
            )
        aggregate_raw_R2_maps(missing_means)
    else:
        print(f"Aggregated raw-R² maps already exist for layers {list(required_layers)}; reusing them.")

    missing_masks = [
        layer for layer in range(1, N_LAYERS + 1)
        if not os.path.exists(contour_mask_path(layer))
    ]
    if missing_masks:
        if args.layer is not None or args.assemble:
            raise FileNotFoundError(
                f"Strict render/assembly refused: missing NSD FDR masks {missing_masks}"
            )
        print(f"Missing NSD FDR masks for layers {missing_masks}; rebuilding them.")
        save_fdr_mask_niis()

    for layer in required_layers:
        validate_mean_and_contour(layer)

    common = dict(
        resultType="NSD nonresidual: raw R2 unthresholded (FDR outline)",
        cmap="RdBu_r",
        vmax=VMAX,
        displayThreshold=None,
        bgDarkness=0.3,
        contourMaskPathFn=contour_mask_path,
        views=["posterior"],
        figureName="raw_R2_surface_outlined_nsd_posterior_",
        copyTo=OUT_PDF,
        colorBarVmin=-VMAX,
        colorBarLabel="raw cross-validated $R^2$",
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
    print(f"Saved NSD raw-R² surface figure: {OUT_PDF}")


if __name__ == "__main__":
    main()
