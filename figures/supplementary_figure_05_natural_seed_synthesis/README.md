# Figure S5: Image synthesis from natural-image seeds

- Panel A shows the 16 natural images used as seeds.
- Panel B shows the final images after cross-layer orthogonalization.
- Panel C shows correlations among the 12 layer-specific RSM fingerprints.

`analysis/replay_fig1d_noise.py` runs the Figure 1 optimization with a supplied
16-image seed manifest. The preflight and validation scripts protect outputs
from overwrite and validate checkpoints. `finalize_sfig5_nsd_seed_run.py`
performs exact CNN forward passes on the final PNGs and derives Panel C.

Seed images, attribution records, the Inception-V1 graph, and the completed
synthesis run are external inputs. Manuscript-PDF assembly is excluded.
