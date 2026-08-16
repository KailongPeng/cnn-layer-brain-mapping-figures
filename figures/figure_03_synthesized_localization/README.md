# Figure 3 — Localization from synthesized images

This directory contains the synthesized-image searchlight analysis used for
the main-text Figure 3.

## Panels

- **A:** Whole-brain leave-one-participant-out layer maps, restricted to the
  representational-stability mask and shown after FDR correction.
- **B:** A consolidated best-layer map assigning each voxel to the layer with
  the largest statistic.

## Relation to the paper

The analysis follows the *Mapping model layers in the brain* subsection of the
Methods in `main_tex.md`. CNN layer RSMs are compared with brain RSMs in a
searchlight using leave-one-participant-out regression. The corrected
brain-RSM permutation is a row/column permutation of the 16-image brain RSM;
it is not a direct shuffle of the 120 upper-triangle entries.

## Inputs and outputs

Inputs are the 12 CNN-layer RSMs, synthesized-image brain representations, and
the stability mask. The analysis produces one statistic map per layer and a
best-layer index map. Subject data and intermediate volumetric products are
not part of this repository.
