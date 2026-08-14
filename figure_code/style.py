from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def prepare_output(path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
