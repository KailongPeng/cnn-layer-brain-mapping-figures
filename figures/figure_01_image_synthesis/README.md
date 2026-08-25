# Figure 1: Image synthesis process

Figure 1 follows one set of 16 images through the synthesis procedure and
compares the resulting layer representations with natural images from NSD.

## Panels

- A shows the random-noise starting images.
- B shows the 16 by 16 image RSM for each CNN layer.
- C shows the second-order correlation matrix across the 12 layer RSMs.
- D shows the final synthesized images and their second-order matrix.
- E and F show the natural-image control and one NSD image set.
- G and H compare synthesized images with 5,000 randomly sampled NSD sets.

## Code

`analysis/ImageSynthesis.py` is the original TensorFlow 1 synthesis program.
`analysis/ImageSynthesis_gpu.py` is its GPU variant. Their shell files are the
Milgram entry points used by the project.

`analysis/baselineLayerCorr_NSD_parent.py` samples the 5,000 sets of 16 NSD
images, submits the per-set calculation, and aggregates the result.
`analysis/baseline.py` performs the CNN forward pass and RSM calculation for a
single set. The local copy of `utils.py` is the helper module imported by the
historical scripts.

The synthesis scripts expect the original Inception graph,
`16corr_GradOrth.npy`, `bestchans16.pkl`, and `corrsvals16.pkl` in `assets/`.
Those assets are not silently regenerated.
`assets/selected_channels_16.json` is a readable record of the selected
channels; the historical script still reads the original pickle. The
TensorFlow 1 environment is recorded in `environment.yml`.

The code produces the images, RSMs, second-order correlations, and control
summaries used for the individual panels. Manuscript PDF assembly is not
included.
