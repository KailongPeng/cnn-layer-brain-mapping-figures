# CNN layer brain mapping figures

This repository contains the image-synthesis and panel-plotting code for the
paper "Synthesizing images to map neural networks to the human brain." Plotting
scripts start from derived NIfTI maps or NumPy arrays. They do not preprocess
fMRI data or assemble panels into manuscript figures.

## Figure-centered layout

The main analysis is being migrated into one directory per main-text figure:

| Directory | Main-text figure |
| --- | --- |
| `figures/figure_01_image_synthesis/` | Figure 1 |
| `figures/figure_02_mapping_schematic/` | Figure 2 |
| `figures/figure_03_synthesized_localization/` | Figure 3 |
| `figures/figure_04_nsd_localization/` | Figure 4 |
| `figures/figure_05_cross_layer_overlap/` | Figure 5 |
| `figures/figure_06_residual_localization/` | Figure 6 |

Each directory has a public `README.md` describing its panels and method.
Local provenance is kept in ignored `_private/` and `.lcp_source/` directories;
these are not required to run the public code. Supplementary figures remain
outside this migration.

## Image synthesis

The TensorFlow 1 program used to synthesize the 16-image stimulus set is in
[`image_synthesis/`](image_synthesis/). Its README describes the historical
environment, required model file, archived target matrix, and run settings.

## Scripts

| Script | Manuscript panels |
| --- | --- |
| `surface_layer_maps.py` | Figures 3A, 4A, 6A, S7, S10-S12, S14, S16, and S17 |
| `winner_layer_map.py` | Figures 3B, 4B, and 6B |
| `correlation_matrix.py` | Figures 5A-B, 6C, S4B-C, S5C, S8A, and S9 |
| `bootstrap_histogram.py` | Figures 5C-D and 6D-E |
| `n_images.py` | Figure S3 |
| `nsd_synthetic.py` | Figure S4 |
| `image_montage.py` | Figure S5A-B |
| `noise_ceiling_roi.py` | Figure S6C |
| `stability_scatter.py` | Figure S6D |
| `split_half.py` | Figure S8 |
| `geodesic_distance.py` | Figure S13 |
| `voxel_counts.py` | Figure S15 |
| `toy_manifold.py` | Figure S18 |

`surface_layer_maps.py` and `winner_layer_map.py` take one NIfTI map per model
layer. The remaining scripts take compact NPY or NPZ files. Run a script with
`--help` for its expected fields and command-line arguments.

Figure 2 is a method schematic and has no standalone numerical generator.
The legacy `figure_code/` scripts remain available during the migration, while
the `figures/` tree is the source of truth for the main-figure organization.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Example:

```bash
python figure_code/correlation_matrix.py \
  --matrix data/synthesized_layer_correlations.npy \
  --output outputs/fig5_matrix_synthesized.pdf \
  --vmin -1 --vmax 1
```

The scripts create the parent directory of each requested output file. Data and
rendered outputs are ignored by Git.
