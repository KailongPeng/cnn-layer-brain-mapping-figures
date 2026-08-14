# Image synthesis

`ImageSynthesis.py` contains the TensorFlow 1 implementation used to create the
16 synthesized images. It uses the twelve Inception V1 layers reported in the
paper. The optimization combines two terms: selected-channel activation and the
sum of squared correlations between layer RSMs. Each gradient has its own Adam
state and Laplacian normalization. Updates also use random foveated crops,
clipping, amplitude decay, and four image scales.

The public copy keeps the experimental settings from the original program. The
filesystem setup has been changed so that it can run outside the lab project.
CPU and GPU execution use the same script.

## Environment

The synthesis environment is separate from the plotting environment because the
original analysis used TensorFlow 1.14 and Python 3.7.

```bash
conda env create -f image_synthesis/environment.yml
conda activate cnn-layer-image-synthesis
```

## Required assets

Follow [`assets/README.md`](assets/README.md) to obtain the Inception graph. Copy
the archived target matrix from the original experiment to:

```text
image_synthesis/assets/16corr_GradOrth.npy
```

Then run the asset gate:

```bash
python image_synthesis/validate_assets.py
```

The repository tracks `selected_channels_16.json` instead of the original
`bestchans16.pkl`. The values are unchanged, but the JSON file can be inspected
without loading a pickle.

## Run

A noise-seeded run uses:

```bash
SYNTH_TAG=noise_seed \
python image_synthesis/ImageSynthesis.py
```

The default device is `/cpu:0`. If TensorFlow 1.14 was installed with compatible
GPU and CUDA support, set `SYNTH_DEVICE=/gpu:0`. Results are written to
`outputs/image_synthesis/<tag>/` by default. `SYNTH_OUTPUT_DIR` can place them
elsewhere.

For the structured-seed control, create a text file containing one image path
per line and run:

```bash
SEED_MODE=structured \
SEED_MANIFEST=/path/to/seed_manifest.txt \
SYNTH_DEVICE=/gpu:0 \
SYNTH_TAG=structured_seed \
python image_synthesis/ImageSynthesis.py
```

`SMOKE_TEST=1` checks initialization and saves the starting images without
running the full optimization.
