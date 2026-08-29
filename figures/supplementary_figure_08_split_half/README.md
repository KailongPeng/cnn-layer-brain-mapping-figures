# Figure S8: Split-half reliability across participants

Thirty participants are randomly divided into two groups of 15 on each of 100
splits. Spatial maps from the two halves are correlated within and between all
12 model layers.

`analysis/analysis_12_split_half_subjects.py` computes the split-level matrices
from corrected-permutation subject outputs. The figure script renders their
mean matrix and within-layer diagonal. Error bars are standard deviations
across splits; the off-diagonal distribution is the between-layer reference.
