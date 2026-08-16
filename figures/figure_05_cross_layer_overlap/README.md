# Figure 5 — Cross-layer overlap

This directory contains the continuous cross-layer overlap analysis used for
the main-text Figure 5.

## Panels

- **A:** Cross-layer Pearson correlations between synthesized-image
  unthresholded effect-size maps.
- **B:** The same correlation matrix for the 16 NSD image sets.
- **C:** Bootstrap comparison of the mean off-diagonal correlation.
- **D:** The corresponding comparison restricted to adjacent layers.

## Relation to the paper

The analysis follows the *Overlap analysis* subsection of the Methods in
`main_tex.md`. It correlates the complete unthresholded whole-brain maps,
rather than only significant voxels. The NSD null distribution is built from
per-set correlations and bootstrap resampling across the 16 image sets.

## Inputs and outputs

Inputs are the 12 unthresholded layer maps for synthesized images and the 16
NSD sets. Outputs are the two correlation matrices, bootstrap distributions,
and summary statistics used by panels A–D.
