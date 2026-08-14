#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

from style import plt, prepare_output, set_style


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def main():
    parser = argparse.ArgumentParser(description="Plot a panel containing an image set.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--output", required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--maximum-images", type=int, default=16)
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob(args.pattern), key=natural_key)
    paths = paths[: args.maximum_images]
    if not paths:
        raise FileNotFoundError(f"No images match {args.pattern!r} in {args.input_dir}")
    if args.columns < 1:
        raise ValueError("columns must be positive")

    rows = int(np.ceil(len(paths) / args.columns))
    set_style()
    fig, axes = plt.subplots(rows, args.columns, figsize=(2.4 * args.columns, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for index, ax in enumerate(axes):
        ax.axis("off")
        if index < len(paths):
            with Image.open(paths[index]) as image:
                ax.imshow(np.asarray(image.convert("RGB")))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)
    fig.savefig(prepare_output(args.output), bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
