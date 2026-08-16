#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

from style import plt, prepare_output, set_style


COLORS = [
    "#ff1744",
    "#ff6d00",
    "#ffd600",
    "#76ff03",
    "#00e676",
    "#00e5ff",
    "#2979ff",
    "#304ffe",
    "#651fff",
    "#aa00ff",
    "#d500f9",
    "#f500c8",
]


def main():
    parser = argparse.ArgumentParser(
        description="Render the layer with the largest positive value at each voxel."
    )
    parser.add_argument("--map-template", required=True, help="NIfTI path with {layer}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--mesh", default="fsaverage5")
    args = parser.parse_args()

    import nibabel as nib
    from nilearn import datasets, plotting, surface

    paths = [Path(args.map_template.format(layer=i)) for i in range(1, args.layers + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing layer maps:\n" + "\n".join(missing))

    images = [nib.load(str(path)) for path in paths]
    reference = images[0]
    for path, image in zip(paths[1:], images[1:]):
        if image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, atol=1e-5
        ):
            raise ValueError(f"NIfTI grid mismatch in {path}")

    values = np.stack([image.get_fdata() for image in images])
    if not np.all(np.isfinite(values)):
        raise ValueError("Layer maps contain nonfinite values")
    maximum = np.max(values, axis=0)
    winner = np.argmax(values, axis=0).astype(float) + 1
    winner[maximum <= 0] = 0
    winner_image = nib.Nifti1Image(winner, reference.affine, reference.header)

    set_style()
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)
    cmap = ListedColormap(COLORS[: args.layers])
    bounds = np.arange(0.5, args.layers + 1.5)
    norm = BoundaryNorm(bounds, cmap.N)
    views = ("lateral", "medial", "posterior")
    fig = plt.figure(figsize=(12.5, 7.0))

    for view_index, view in enumerate(views):
        for hemi_index, hemi in enumerate(("left", "right")):
            ax = fig.add_subplot(2, 3, view_index + 1 + 3 * hemi_index, projection="3d")
            pial = getattr(fsaverage, f"pial_{hemi}")
            inflated = getattr(fsaverage, f"infl_{hemi}")
            sulc = getattr(fsaverage, f"sulc_{hemi}")
            texture = surface.vol_to_surf(
                winner_image, pial, interpolation="nearest_most_frequent"
            )
            plotting.plot_surf_stat_map(
                inflated,
                texture,
                hemi=hemi,
                view=view,
                bg_map=sulc,
                bg_on_data=True,
                cmap=cmap,
                vmin=0.5,
                vmax=args.layers + 0.5,
                threshold=0.5,
                colorbar=False,
                axes=ax,
            )
            ax.set_axis_off()

    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.62])
    ticks = [1, 3, 6, 9, args.layers]
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, ticks=ticks)
    colorbar.set_label("Maximum layer representation", fontsize=16, labelpad=15)
    colorbar.ax.tick_params(labelsize=12)
    fig.subplots_adjust(left=0.01, right=0.89, bottom=0.01, top=0.99, wspace=-0.25, hspace=-0.25)
    fig.savefig(prepare_output(args.output), bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
