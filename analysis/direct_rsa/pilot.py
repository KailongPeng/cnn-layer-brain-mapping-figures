"""Small explicit-location RSA checks. Produces JSON, never a partial brain map."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np

from .core import direct_pearson


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--n-centers', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.subject not in cfg['subjects'] or args.n_centers < 1:
        raise ValueError('Invalid subject or sample count')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Preserve existing pilot result: {output}')
    mask_path = cfg['mask'].format(subject=args.subject)
    mask_image = nib.load(mask_path)
    mask = mask_image.get_fdata() > 0
    images = []
    input_records = []
    for i in range(1, 17):
        path = Path(cfg['cope'].format(subject=args.subject, image=i))
        img = nib.load(str(path))
        if img.shape != mask_image.shape or not np.allclose(img.affine, mask_image.affine, atol=1e-4):
            raise ValueError(f'Cope/mask spatial mismatch: {path}')
        images.append(img.get_fdata())
        input_records.append({'path': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns})
    data = np.stack(images, axis=-1)
    features = np.load(cfg['model_rsms'], allow_pickle=False)
    if features.shape != (12, 16, 16):
        raise ValueError('Unexpected model RSM shape')
    radius = int(cfg['radius'])
    interior = mask.copy()
    for axis in range(3):
        front = [slice(None)] * 3; front[axis] = slice(0, radius)
        back = [slice(None)] * 3; back[axis] = slice(-radius, None)
        interior[tuple(front)] = False; interior[tuple(back)] = False
    coords = np.argwhere(interior)
    coords = coords[np.random.default_rng(args.seed).permutation(len(coords))]
    points = []
    skipped = 0
    for coord in coords:
        sl = tuple(slice(int(v)-radius, int(v)+radius+1) for v in coord)
        patterns = data[sl].reshape(-1, 16).T
        values = direct_pearson(features, patterns)
        if not np.isfinite(values).all():
            skipped += 1
            continue
        half = direct_pearson(features, patterns, indices=list(range(0,16,2)))
        points.append({'ijk': coord.tolist(), 'r_16_images': values.tolist(),
                       'r_8_images': [float(v) if np.isfinite(v) else None for v in half]})
        if len(points) == args.n_centers:
            break
    if len(points) != args.n_centers:
        raise RuntimeError('Not enough valid searchlights')
    result = {
        'subject': args.subject, 'pilot_only': True, 'n_centers': len(points),
        'radius': radius, 'searchlight': 'full cube; full-cube image patterns, center selected by mask',
        'image_order_status': cfg['image_order_status'], 'skipped_undefined_centers': skipped,
        'model_sha256': sha256(cfg['model_rsms']), 'config_sha256': sha256(args.config),
        'cope_inputs': input_records, 'points': points,
        'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'subject': args.subject, 'centers': len(points), 'skipped': skipped,
                      'output': str(output), 'pilot_only': True}))


if __name__ == '__main__':
    main()
