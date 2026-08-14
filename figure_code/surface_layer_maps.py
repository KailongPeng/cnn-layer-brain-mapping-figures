#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from style import plt, prepare_output, set_style


def layer_path(template, layer):
    path = Path(template.format(layer=layer))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_images(paths):
    import nibabel as nib

    images = [nib.load(str(path)) for path in paths]
    shape = images[0].shape
    affine = images[0].affine
    for path, image in zip(paths[1:], images[1:]):
        if image.shape != shape:
            raise ValueError(f"Shape mismatch in {path}: {image.shape} != {shape}")
        if not np.allclose(image.affine, affine, atol=1e-5):
            raise ValueError(f"Affine mismatch in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render one panel containing the cortical map for each layer."
    )
    parser.add_argument(
        "--map-template",
        required=True,
        help="NIfTI path containing {layer}, for example data/layer{layer}.nii.gz",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--view", default="posterior")
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", required=True, type=float)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--colorbar-label", default="t statistic value")
    parser.add_argument("--contour-template", default=None)
    parser.add_argument("--mesh", default="fsaverage5")
    args = parser.parse_args()

    import nibabel as nib
    from nilearn import datasets, plotting, surface

    if args.layers < 1 or args.vmax <= args.vmin:
        raise ValueError("Invalid number of layers or color range")

    map_paths = [layer_path(args.map_template, i) for i in range(1, args.layers + 1)]
    check_images(map_paths)
    contour_paths = None
    if args.contour_template:
        contour_paths = [
            layer_path(args.contour_template, i) for i in range(1, args.layers + 1)
        ]
        check_images(contour_paths)

    set_style()
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)
    rows = int(np.ceil(args.layers / 4))
    fig = plt.figure(figsize=(15.2, 3.8 * rows))

    for layer, map_path in enumerate(map_paths, start=1):
        row = (layer - 1) // 4
        column = (layer - 1) % 4
        for hemi_index, hemi in enumerate(("left", "right")):
            index = row * 8 + column * 2 + hemi_index + 1
            ax = fig.add_subplot(rows, 8, index, projection="3d")
            pial = getattr(fsaverage, f"pial_{hemi}")
            inflated = getattr(fsaverage, f"infl_{hemi}")
            sulc = getattr(fsaverage, f"sulc_{hemi}")
            texture = surface.vol_to_surf(str(map_path), pial, interpolation="linear")
            plotting.plot_surf_stat_map(
                inflated,
                texture,
                hemi=hemi,
                view=args.view,
                bg_map=sulc,
                bg_on_data=True,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                threshold=args.threshold,
                colorbar=False,
                axes=ax,
            )
            if contour_paths:
                contour = surface.vol_to_surf(
                    str(contour_paths[layer - 1]),
                    pial,
                    interpolation="nearest_most_frequent",
                )
                plotting.plot_surf_contours(
                    inflated,
                    contour,
                    hemi=hemi,
                    view=args.view,
                    levels=[0.5],
                    colors=["black"],
                    axes=ax,
                )
            ax.set_axis_off()
        x = (column * 2 + 1) / 8
        y = 1.0 - row / rows - 0.018
        fig.text(x, y, str(layer), ha="center", va="top", fontsize=17)

    scalar = ScalarMappable(
        norm=Normalize(vmin=args.vmin, vmax=args.vmax), cmap=args.cmap
    )
    scalar.set_array([])
    colorbar_ax = fig.add_axes([0.925, 0.25, 0.014, 0.5])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label(args.colorbar_label, fontsize=16, rotation=90, labelpad=15)
    colorbar.ax.tick_params(labelsize=12)
    fig.subplots_adjust(left=0.01, right=0.91, bottom=0.01, top=0.97, wspace=-0.22, hspace=-0.1)
    fig.savefig(prepare_output(args.output), bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
