# Figure 6 — Residual localization from natural images

This directory contains the residual-method analysis used for the main-text
Figure 6.

## Panels

- **A:** Residualized whole-brain layer maps averaged across the 16 NSD sets.
- **B:** The corresponding best-layer index map.
- **C:** Cross-layer correlations after residualization.
- **D:** Bootstrap comparison across all layer pairs.
- **E:** The same comparison restricted to adjacent layers.

## Relation to the paper

The analysis follows the *Residual method* subsection of the Methods in
`main_tex.md`. For each held-out CNN layer and searchlight, the other 11 layer
RSMs are used in a multiple regression. The remaining residual variance is
then related to the held-out layer RSM in a leave-one-participant-out
procedure.

## Inputs and outputs

Inputs are NSD image-wise brain representations, the 16-set manifest, CNN
layer RSMs, and the stability mask. Outputs are residual layer maps,
best-layer maps, cross-layer correlations, and bootstrap summaries. The exact
held-out-target convention is recorded alongside the migrated analysis code
and must not be changed silently during refactoring.
