# Figure S18: Toy-manifold illustration

The analysis generates paired S-shaped and unrolled manifolds, selects 64
corresponding points, and iteratively relocates them to minimize the squared
correlation between the two layer distance matrices. Panels compare point
locations before and after optimization in both manifolds.

`analysis/swissRoll.py` contains the simulation, optimization, and panel-level
plots. This conceptual analysis does not use participant or NSD data.
