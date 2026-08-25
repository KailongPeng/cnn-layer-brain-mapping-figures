# Figure 6: Residual NSD localization

Figure 6 applies the residual method to the same 16 NSD image sets used in
Figure 4. For a target layer, the other 11 layer RSMs first predict the brain
RSM. The target layer is then fitted to the remaining brain variance.

## Panels

- A contains the FDR-corrected residual map for each layer.
- B assigns each eligible voxel to the layer with the largest t statistic.
- C is the cross-layer correlation matrix for the residual maps.
- D and E compare residual and non-residual NSD correlations with bootstrap
  resampling across image sets.

## Analysis order

1. `analysis/searchlight_train_linearModel/` fits the residual model in a
   leave-one-participant-out analysis.
2. `analysis/searchlight_score_linearModel/` scores the held-out participant.
3. `analysis/nsd_allSetMeanRandomise/` averages the 16 sets within participant
   and prepares the group result.
4. `analysis/fig6ab/fig6a_fdr.py` and `fig6b_fdr.py` generate panels A and B.
5. `analysis/analysis_02_raw_R2_maps.py`,
   `analysis/analysis_04_continuous_overlap.py`,
   `analysis/analysis_04b_continuous_overlap_bootstrap.py`, and
   `analysis/viz_fig6_overlap_continuous.py` generate panels C through E.

The figure directory includes its own copies of `utils.py`, `config.py`, the
NIfTI validation helper, and `display_surface_plot/`. The files retain the
Milgram paths and the held-out scoring convention used in the original
analysis.

The permutation null is formed by applying one permutation of the 16 stimulus
labels to both axes of the held-out brain RSM. The upper triangle is extracted
and standardized only after this coordinated reordering, so the shuffled data
remain valid RSMs.
