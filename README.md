# CNN layer brain mapping figures

This repository contains the image-synthesis and panel-plotting code for the
paper "Synthesizing images to map neural networks to the human brain." Plotting
scripts start from derived NIfTI maps or NumPy arrays. They do not preprocess
fMRI data or assemble panels into manuscript figures.

No subject data, model checkpoints, intermediate searchlight files, rendered
figures, or cluster job files are included.

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

Figures 1, 2, S1, and S2 are method or task schematics. They are not generated
from analysis arrays and have no plotting script in this repository.

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
