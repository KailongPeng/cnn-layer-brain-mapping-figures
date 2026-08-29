# Figure S9: Thresholded overlap of searchlight maps

This robustness analysis treats each corrected layer map as a binary set of
significant voxels. For ordered layer pairs M and N, overlap is the number of
shared voxels divided by the number of voxels in M, so the matrix need not be
symmetric.

`analysis/fig5b.py` computes the synthesized-image matrix.
`analysis/nsd_overlap/nsd_overlapBetweenSets.py` computes 16 NSD set-level
matrices and 5,000-sample bootstrap comparisons for all and adjacent layers.
Figure 5 instead uses correlations of continuous, unthresholded maps.
