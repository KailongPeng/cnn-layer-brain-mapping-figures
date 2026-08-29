# Figure S4: NSD-Synthetic distinguishability

This analysis compares cross-layer RSM correlations for the 16 optimized
stimuli, the first 16 NSD-Synthetic stimuli, and 100 resampled 16-image subsets
of NSD-Synthetic.

`analysis/analysis_09b_nsd_synthetic.py` performs the CNN forward passes,
bootstrap calculation, and matrix analysis. The `job_09b_*` files are the
historical Slurm entry points. `restyle_sfig11_nsd_synthetic.py` regenerates the
panel output from the analysis cache. The NSD-Synthetic stimulus file and
Inception-V1 graph are external inputs.
