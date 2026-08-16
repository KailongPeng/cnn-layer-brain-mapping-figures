# Figure 4 — Localization from natural images

This directory contains the non-residual NSD searchlight analysis used for the
main-text Figure 4.

## Panels

- **A:** Whole-brain leave-one-participant-out layer maps averaged across the
  16 NSD image sets and restricted to the stability mask.
- **B:** The corresponding best-layer index map.

## Relation to the paper

The analysis follows the natural-image part of *Mapping model layers in the
brain* in the Methods in `main_tex.md`. For each participant, layer maps are
computed separately for the 16 image sets and then averaged within participant
before group inference.

## Inputs and outputs

Inputs are NSD image-wise brain representations, the 16-set manifest, CNN
layer RSMs, and the NSD stability mask. Outputs are layer-specific statistic
maps and a consolidated best-layer map. Raw NSD data and intermediate
searchlight volumes are not included here.
