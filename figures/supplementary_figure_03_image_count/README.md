# Figure S3: Effect of image count on cross-layer RSM estimates

The 16 synthesized images are resampled 500 times at each subset size from 4
through 16. The script computes the mean and standard deviation of the
off-diagonal cross-layer RSM correlations. The mean remains near zero while
sampling variability decreases as more images are included.

Run `analysis/viz_n_images_analysis.py` with the synthesized-image CNN RSMs at
the historical `features.npy` path. This is a CNN-representation analysis and
does not rerun the fMRI searchlight.
