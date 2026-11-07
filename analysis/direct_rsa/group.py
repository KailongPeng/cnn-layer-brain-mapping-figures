"""Descriptive subject-level Fisher aggregation, without significance claims."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np


def fisher_mean(values):
    """Average finite subject correlations, retaining missing coverage as NaN."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if np.any(np.abs(values[valid]) > 1):
        raise ValueError('Input outside Pearson range')
    count = valid.sum(axis=0)
    epsilon = np.finfo(np.float32).eps
    z = np.zeros(values.shape, dtype=np.float64)
    z[valid] = np.arctanh(np.clip(values[valid], -1+epsilon, 1-epsilon))
    mean_z = np.full(count.shape, np.nan)
    np.divide(z.sum(axis=0), count, out=mean_z, where=count > 0)
    return np.tanh(mean_z), count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    source, output = Path(args.input_dir), Path(args.output_dir)
    if output.exists():
        raise FileExistsError(str(output))
    subjects = cfg['subjects']
    if len(subjects) != 30 or len(set(subjects)) != 30:
        raise ValueError('Require the configured 30 unique subjects')
    model_hash = hashlib.sha256(Path(cfg['model_rsms']).read_bytes()).hexdigest()
    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    manifests = []
    for subject in subjects:
        manifest = json.loads((source/subject/'qc.json').read_text())
        if (manifest['subject'] != subject or manifest['validation_only'] or
                manifest.get('image_indices') != list(range(16)) or
                manifest.get('model_sha256') != model_hash or
                manifest.get('config_sha256') != config_hash):
            raise ValueError('Input provenance mismatch: '+subject)
        manifests.append(manifest)
    reference = None
    means, counts, layer_qc = [], [], []
    for layer in range(1, 13):
        arrays = []
        for subject in subjects:
            image = nib.load(str(source/subject/('layer%02d_r.nii.gz' % layer)))
            if reference is None:
                reference = image
            if (image.shape != reference.shape or
                    not np.allclose(image.affine, reference.affine, atol=1e-4)):
                raise ValueError('Spatial mismatch: '+subject)
            arrays.append(image.get_fdata(dtype=np.float32))
        mean, count = fisher_mean(np.asarray(arrays))
        means.append(mean.astype(np.float32)); counts.append(count.astype(np.uint8))
        layer_qc.append({'layer': layer, 'any_coverage_voxels': int((count>0).sum()),
                         'all_30_voxels': int((count==30).sum()),
                         'r_min': float(np.nanmin(mean)), 'r_max': float(np.nanmax(mean))})
        print('layer %d complete' % layer, flush=True)
    common = np.logical_and.reduce([count==30 for count in counts])
    if common.sum() < 3:
        raise ValueError('Insufficient common coverage')
    spatial = np.corrcoef(np.asarray([mean[common] for mean in means], dtype=np.float64))
    output.mkdir(parents=True, exist_ok=False)
    for layer, (mean, count) in enumerate(zip(means, counts), 1):
        nib.save(nib.Nifti1Image(mean, reference.affine), str(output/('layer%02d_mean_r.nii.gz'%layer)))
        nib.save(nib.Nifti1Image(count, reference.affine), str(output/('layer%02d_n.nii.gz'%layer)))
    nib.save(nib.Nifti1Image(common.astype(np.uint8), reference.affine), str(output/'common_all30_mask.nii.gz'))
    np.savetxt(str(output/'spatial_correlation_common_all30.csv'), spatial, delimiter=',')
    report = {'descriptive_only': True, 'subjects': subjects, 'n_subjects': len(subjects),
              'config_sha256': config_hash, 'model_sha256': model_hash,
              'aggregation': 'tanh(mean(atanh(subject r))) over finite subjects',
              'endpoint_clip': float(np.finfo(np.float32).eps),
              'spatial_comparison_mask': 'all 30 subjects finite at all 12 layers; no significance selection',
              'common_all30_voxels': int(common.sum()), 'layer_qc': layer_qc,
              'subject_code_commits': sorted(set(m['code_commit'] for m in manifests)),
              'code_commit': subprocess.check_output(['git','rev-parse','HEAD'], universal_newlines=True).strip()}
    (output/'qc.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
