# Figure S12: Cluster-mass correction

This figure repeats localization with cluster-mass correction and a
cluster-forming threshold of p < 0.001 instead of the FDR procedure used in the
main figures.

- `analysis/synthesized/` contains the Figure 3 alternative correction.
- `analysis/nsd_nonresidual/` contains the Figure 4 alternative correction.
- `analysis/nsd_residual/` contains the Figure 6 alternative correction.

`analysis/Sfig4.py` is the historical dispatcher. These analyses consume
participant-level searchlight maps and do not rerun model fitting.
