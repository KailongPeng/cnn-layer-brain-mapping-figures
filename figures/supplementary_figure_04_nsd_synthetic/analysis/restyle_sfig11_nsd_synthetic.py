#!/usr/bin/env python3
"""Compatibility entry point for exact Figure S11 regeneration.

The earlier implementation restyled an already rendered PDF. That route cannot
change an asymmetric color normalization without risking a mismatch between
matrix colors and colorbar values. The supported path now loads the compact,
git-tracked NPY-derived cache and regenerates the figure from the matrices.
"""

from analysis_09b_nsd_synthetic import run_tracked_plot_cache


if __name__ == "__main__":
    run_tracked_plot_cache()
