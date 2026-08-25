# Figure 3: Synthesized-image localization

Figure 3 maps each of the 12 CNN layer RSMs to searchlight RSMs measured while
participants viewed the synthesized images.

## Panels

- A contains the FDR-corrected posterior surface map for each layer.
- B assigns each eligible voxel to the layer with the largest t statistic.

## Analysis order

1. `analysis/stability/` computes the synthesized-image representational
   stability map used as the analysis mask.
2. `analysis/searchlight_train_linearModel/` fits the leave-one-participant-out
   regression coefficients.
3. `analysis/corrected_permutation/` scores held-out participants with the
   coordinated 16 by 16 RSM label permutation used for the final null.
4. `analysis/fig3a_fdr.py` and `analysis/fig3b_fdr.py` generate the two panel
   results. `analysis/viz_tstat_outlined_synthesized.py` produces the
   unthresholded surface rendering used to check the same statistics.

The copied `utils.py`, `config.py`, and `display_surface_plot/` module are kept
inside this figure directory because these scripts imported them in the
original project. The source retains the Milgram project and scratch paths.

Expected inputs include image-wise participant beta maps, the 12 CNN RSMs,
and the synthesized-image stability mask. Outputs are participant score maps,
group t maps, FDR masks, and the panel-level surface images.
