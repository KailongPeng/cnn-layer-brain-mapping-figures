#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_LAYERS = (
    'conv2d0_pre_relu',
    'conv2d1_pre_relu',
    'conv2d2_pre_relu',
    'mixed3a_pool_reduce_pre_relu',
    'mixed3b_pool_reduce_pre_relu',
    'mixed4a_pool_reduce_pre_relu',
    'mixed4b_pool_reduce_pre_relu',
    'mixed4c_pool_reduce_pre_relu',
    'mixed4d_pool_reduce_pre_relu',
    'mixed4e_pool_reduce_pre_relu',
    'mixed5a_pool_reduce_pre_relu',
    'mixed5b_pool_reduce_pre_relu',
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Check the files required for image synthesis.')
    parser.add_argument(
        '--assets',
        type=Path,
        default=Path(__file__).resolve().parent / 'assets',
    )
    args = parser.parse_args()

    channels_path = args.assets / 'selected_channels_16.json'
    target_path = args.assets / '16corr_GradOrth.npy'
    graph_path = args.assets / 'tensorflow_inception_graph.pb'
    for path in (channels_path, target_path, graph_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing synthesis asset: {path}')

    with open(channels_path, 'r') as f:
        channels = json.load(f)
    if tuple(channels) != EXPECTED_LAYERS:
        raise ValueError('The selected-channel keys or their order do not match the 12 model layers')
    for layer, values in channels.items():
        values = np.asarray(values)
        if values.shape != (16,) or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
            raise ValueError(f'Invalid selected channels for {layer}: {values}')

    target = np.load(target_path, allow_pickle=False)
    if target.shape != (12, 16, 16) or not np.all(np.isfinite(target)):
        raise ValueError(f'Expected a finite 12 x 16 x 16 target matrix, got {target.shape}')

    print(f'selected channels  {sha256(channels_path)}')
    print(f'target matrix      {sha256(target_path)}')
    print(f'Inception graph    {sha256(graph_path)}')
    print('SYNTHESIS ASSET GATE PASSED')


if __name__ == '__main__':
    main()
