"""Audit legacy direct-r maps against the current pilot, without overwriting them."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--pilot-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(f'Preserve existing audit: {output}')
    missing=[]; header_issues=[]; comparisons=[]; found=0
    radius=int(cfg['radius'])
    for sub in cfg['subjects']:
        reference=nib.load(cfg['mask'].format(subject=sub))
        for layer in range(1,13):
            path=Path(cfg['legacy_result'].format(subject=sub,radius=radius,layer=layer))
            if not path.is_file():
                missing.append(str(path)); continue
            img=nib.load(str(path)); found+=1
            if img.shape!=reference.shape or not np.allclose(img.affine,reference.affine,atol=1e-4):
                header_issues.append(str(path))
        pilot=Path(args.pilot_dir)/f'{sub}.json'
        if not pilot.exists():
            continue
        d=json.loads(pilot.read_text())
        for variant in ('original','noNAN'):
            diffs=[]; total=0; old_zero=0; out_of_range=0; nonfinite=0; variant_files=0
            for layer in range(1,13):
                path=Path(cfg['legacy_result'].format(subject=sub,radius=radius,layer=layer))
                if variant=='noNAN':
                    path=path.with_name(path.name.replace('-cor.nii.gz','-cor_noNAN.nii.gz'))
                if not path.exists():
                    continue
                img=nib.load(str(path))
                if img.shape!=reference.shape or not np.allclose(img.affine,reference.affine,atol=1e-4):
                    continue
                old=img.get_fdata(); variant_files+=1
                for point in d['points']:
                    value=old[tuple(point['ijk'])]
                    if not np.isfinite(value):
                        nonfinite+=1; continue
                    total+=1
                    if abs(value)>1+1e-6:
                        out_of_range+=1; continue
                    if value==0:
                        old_zero+=1
                        continue  # invalid legacy voxels and true zero are indistinguishable
                    diffs.append(abs(float(value)-point['r_16_images'][layer-1]))
            comparisons.append({'subject':sub,'variant':variant,'readable_variant_files':variant_files,
                                'finite_pairs':total,'legacy_zero_pairs':old_zero,
                                'legacy_out_of_range_pairs':out_of_range,'legacy_nonfinite_pairs':nonfinite,
                                'valid_nonzero_pairs_compared':len(diffs),
                                'max_absolute_difference':max(diffs) if diffs else None,
                                'mean_absolute_difference':float(np.mean(diffs)) if diffs else None,
                                'all_valid_nonzero_pairs_match_atol_1e-6':bool(np.all(np.array(diffs)<=1e-6)) if diffs else None})
    result={'radius':radius,'expected_maps':len(cfg['subjects'])*12,'found_maps':found,
            'missing':missing,'header_issues':header_issues,'pilot_comparisons':comparisons,
            'interpretation':'Matching headers alone do not verify provenance. Disagreement requires source/processing review before legacy map reuse.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
