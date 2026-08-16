# Figure 1 — Image synthesis process

This directory contains the analysis code for the main-text Figure 1. The
figure explains how a 16-image stimulus set is optimized so that the
representational similarity matrices (RSMs) of 12 Inception-V1 layers become
as distinct as possible.

## Panels

- **A:** 16 visual-noise starting images.
- **B:** Layer-wise 16 × 16 image RSMs.
- **C:** The 12 × 12 second-order correlation matrix across layer RSMs.
- **D:** The optimized synthesized-image result.
- **E–F:** Natural-image control and one NSD image set.
- **G–H:** Layer-correlation summaries for all pairs and adjacent layers.

## Relation to the paper

The implementation follows the *Image synthesis* subsection of the Methods in
`main_tex.md`: image optimization, layer RSM construction, second-order
correlation, and the NSD control distribution. The public code starts from
model weights and the selected-channel/image-set manifests supplied by the
user; it does not include private cluster data or rendered manuscript PDFs.

## Running

The analysis scripts are in `analysis/`. Each script documents its input and
output paths at the command line. Run `python validate_inputs.py --help` before
starting a full optimization. Large model assets and stimulus arrays are
provided separately when their redistribution is permitted.
