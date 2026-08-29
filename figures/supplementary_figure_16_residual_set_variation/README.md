# Figure S16: Residual-method variation across NSD sets

This repeats Figure S14 with the residual method. The 16 separately
thresholded, stability-masked results are summed for each layer to show how
often a voxel is identified across image sets.

`analysis/nsd_allSetAnalysis.py` derives set-count volumes from Figure 6
outputs. `analysis/Sfig6.py` renders those volumes. Residual regression is not
refit in this directory.
