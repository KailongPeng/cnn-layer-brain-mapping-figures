# Figure S13: Best-layer index and distance from V1

For every voxel with a significant synthesized-image mapping, the analysis
selects the layer with the largest group statistic and relates that index to
distance from bilateral V1. Results are summarized in 10-mm bins with SEM.

Run `analysis/analysis_07_geodesic_distance.py` with the Wang atlas and Figure
3 participant maps. The historical code computes a volumetric Euclidean
approximation from V1, although the manuscript calls the measure geodesic;
this implementation detail is preserved explicitly.
