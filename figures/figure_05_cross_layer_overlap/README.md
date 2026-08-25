# Figure 5: Cross-layer correlation

Figure 5 compares the spatial similarity of the 12 whole-brain layer maps for
synthesized images and natural images.

## Panels

- A is the 12 by 12 correlation matrix for synthesized-image maps.
- B is the mean of the 16 per-set NSD correlation matrices.
- C compares the synthesized off-diagonal mean with a 5,000-sample NSD
  bootstrap distribution.
- D repeats the comparison for adjacent layer pairs.

`analysis/analysis_02_raw_R2_maps.py` loads the continuous layer maps.
`analysis/analysis_04_continuous_overlap.py` computes the per-dataset and
per-set matrices inside the representational-stability mask.
`analysis/analysis_04b_continuous_overlap_bootstrap.py` performs the bootstrap
comparison. `analysis/viz_fig5_overlap_continuous.py` renders panels A through
D. The analysis uses complete unthresholded effect maps rather than binary
significance masks.

The source retains the original Milgram result paths and imports a local copy
of `config.py` and `utils.py`. The method decision about averaging per-set NSD
correlation matrices is recorded in
`analysis/0001-per-set-vs-grand-mean-cross-layer-overlap.md`.
