# CNN layer brain mapping figures

This repository contains the analysis code behind the six main figures in
"Synthesizing images to map neural networks to the human brain." The directory
names follow the final figure numbers in the manuscript.

| Directory | Contents |
| --- | --- |
| `figures/figure_01_image_synthesis/` | Image synthesis and the 5,000-set NSD control |
| `figures/figure_02_mapping_schematic/` | Method schematic and its analysis references |
| `figures/figure_03_synthesized_localization/` | Synthesized-image stability and layer mapping |
| `figures/figure_04_nsd_localization/` | Non-residual NSD layer mapping |
| `figures/figure_05_cross_layer_overlap/` | Cross-layer map correlations and bootstrap tests |
| `figures/figure_06_residual_localization/` | Residual NSD mapping and overlap analysis |

The files in each `analysis/` directory are drawn from the historical analysis
scripts. Historical Milgram paths and execution assumptions have been left in
place. Small release corrections are described in the relevant figure README.
This is not a unified Python package, and the same helper may appear in more
than one figure directory when both analyses used it.

Each figure README gives the panel definitions, source scripts, expected
inputs, and main entry points. The repository stops at panel-level output and
does not include the code used to assemble complete manuscript PDFs.

See [ENVIRONMENT.md](ENVIRONMENT.md) for the historical software environments
and external command-line dependencies. Unless a figure README says otherwise,
run entry points from the repository root.

Supplementary figures are not part of this release yet. Local provenance and
migration notes are stored in ignored `_private/` directories.
