# 0001. Report cross-layer overlap at the per-set level, not grand-mean

Status: Accepted (2026-07-19)

## Context

Reviewer R1 Point 5 required replacing the layer-to-layer overlap metric. The original
submission thresholded each layer's whole-brain searchlight map and computed a binary
overlap ratio between the two sets of significant voxels (`mn / (mn + m)`). The revision
replaces this with a continuous measure: the Pearson correlation between two layers'
unthresholded effect-size maps across voxels.

For the NSD and residual datasets there are 16 image sets, and the continuous correlation
can be aggregated over sets in two different orders that do **not** give the same number,
because correlation is nonlinear:

- **grand-mean** — average each layer's 16 maps into one map, then correlate the averaged
  maps once. Averaging removes measurement noise, so the correlation is higher.
  Off-diagonal means: NSD 0.595, residual 0.126.
- **per-set** — correlate each set's maps on its own (16 matrices), then average the
  matrices. Each single-set correlation is attenuated by that set's noise, so the mean is
  lower. Off-diagonal means: NSD 0.245, residual -0.014.

The figure mixes these by construction. A matrix panel (Fig 5B, Fig 6C) shows a single
matrix, which is naturally the grand-mean. A bootstrap panel (Fig 5D/E, Fig 6D/E) needs a
distribution to resample and is therefore necessarily per-set. Reporting grand-mean in the
matrix and text while the bootstrap null is per-set forces a bridging caption note to
explain why the two numbers differ.

The original submission avoided this: its displayed NSD matrix was the element-wise mean of
the 16 per-set overlap matrices (`nsd_overlapBetweenSets.py`, `np.nanmean(..., axis=0)`),
and the bootstrap resampled the off-diagonal means of those same per-set matrices. The
matrix off-diagonal therefore equaled the histogram center exactly. The grand-mean
approach was introduced only in this revision round (`analysis_04` loads the pre-averaged
`nsd_mean` maps), which R1 Point 5 did not ask for.

## Decision

Report cross-layer overlap at the **per-set** level everywhere.

- The displayed matrices (Fig 5B, Fig 6C) are the element-wise mean of the 16 per-set
  correlation matrices. `analysis_04b` saves this as `overlap_{metric}_{dataset}_perset.npy`
  from the same per-set matrices it already builds for the bootstrap.
- The bootstrap nulls stay per-set (unchanged).
- The text reports the per-set numbers: NSD 0.245 / 0.375, residual -0.014 / -0.108,
  synthesized 0.040 / 0.062 (synthesized has a single set, so grand and per-set coincide).
- A fail-loud gate in `analysis_04b` asserts the per-set-averaged matrix's off-diagonal
  mean equals the bootstrap null center, so the matrix and histogram can never silently
  diverge.

Only the per-pair metric changed (binary overlap to correlation), which is what R1 Point 5
required. The averaging order was restored to the original submission's.

## Consequences

- The matrix, the histogram, and the text share one aggregation level. No bridging caption
  note is needed, and the figure matches the internally consistent structure of the
  original submission.
- The reported natural-image correlation (0.245) is lower and less striking than the
  noise-reduced grand-mean (0.595). Grand-mean is arguably the better estimate of the
  "true" denoised cross-layer correlation; we forgo it for internal consistency and for
  fidelity to the original method.
- The residual correlation is slightly negative (-0.014 all pairs, -0.108 adjacent). This
  is expected rather than an error: the residual method removes all variance a layer shares
  with the other 11 layers, so cross-layer correlation is driven to approximately zero and
  can overshoot to slightly negative. This is presented as a limitation of the residual
  method (see reviewer R2 Point 3, and the caveat in the Methods and Discussion), not as
  evidence that the residual method is more selective than synthesis.
- Relatedly, Fig 6D/E was reframed: it now compares the residualized natural images to the
  un-residualized natural images (residualization significantly lowers correlation,
  p < 0.0002), not synthesized versus residual. A synthesized-versus-residual significance
  test is invalid under R2 Point 3, because the two methods are not on a comparable axis.
