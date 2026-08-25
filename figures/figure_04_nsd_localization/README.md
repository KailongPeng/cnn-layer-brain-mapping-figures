# Figure 4: NSD localization

Figure 4 applies the non-residual layer-mapping analysis to 16 sets of 16
natural images from NSD.

## Panels

- A contains the FDR-corrected posterior surface map for each layer.
- B assigns each eligible voxel to the layer with the largest t statistic.

## Analysis order

1. `analysis/nsd_dataPrepare/` extracts the image-wise NSD brain data and CNN
   representations.
2. `analysis/stability/` computes the NSD representational-stability mask.
3. `analysis/nsd_findLayerInBrain_leaveOneSubOutReg/` runs training and scoring
   separately for each image set, then averages the 16 sets within participant.
4. `analysis/fig4a_fdr.py` and `analysis/fig4b_fdr.py` generate the two panel
   results.

The local `utils.py` and `display_surface_plot/` copy are direct dependencies
of the historical scripts. Milgram, NSD, FSL, and scratch paths remain as they
were used for the analysis.

Expected inputs include NSD beta files, the image and repetition index table,
the 12 CNN RSMs for every image set, and the NSD stability mask. Outputs are
per-set participant maps, participant means across sets, group t maps, FDR
masks, and the panel-level surface images.

The scoring worker uses stimulus-label permutation: one permutation of the 16
labels is applied to both axes of the held-out brain RSM before its upper
triangle is scored. This replaces the older element-wise shuffle and preserves
the structure of a valid RSM.
