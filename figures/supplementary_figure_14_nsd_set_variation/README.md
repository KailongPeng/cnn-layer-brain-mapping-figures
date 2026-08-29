# Figure S14: Variation across NSD image sets

The non-residual searchlight and group analysis is run separately for each of
16 NSD image sets. Significant, stability-masked maps are summed so each voxel
records the number of sets in which a layer was identified.

`analysis/nsd_overlapBetweenSets.py` performs set-level masking and summation.
`analysis/Sfig5.py` renders the layer maps. Upstream set-level model fits come
from Figure 4.
