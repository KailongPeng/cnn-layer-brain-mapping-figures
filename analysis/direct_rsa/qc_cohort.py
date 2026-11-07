"""Independently inspect all 360 maps and recompute group estimates voxelwise."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--root',required=True)
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());root=Path(args.root)
    report_path=root/'cohort_independent_qc.json'
    if report_path.exists():raise FileExistsError(str(report_path))
    ref=nib.load(cfg['mask'].format(subject=cfg['subjects'][0]))
    common=np.ones(ref.shape,dtype=bool)
    layer_reports=[];subject_valid={s:np.ones(ref.shape,dtype=bool) for s in cfg['subjects']}
    for layer in range(1,13):
        total=np.zeros(ref.shape);counts=np.zeros(ref.shape,dtype=np.int16)
        for subject in cfg['subjects']:
            path=root/'full'/subject/('layer%02d_r.nii.gz'%layer)
            img=nib.load(str(path))
            if img.shape!=ref.shape or not np.allclose(img.affine,ref.affine,atol=1e-4):
                raise ValueError('Map geometry mismatch: '+str(path))
            data=img.get_fdata();valid=np.isfinite(data)
            if np.isinf(data).any() or np.any(np.abs(data[valid])>1):
                raise ValueError('Invalid Pearson values: '+str(path))
            subject_valid[subject]&=valid
            common&=valid
            counts+=valid
            epsilon=float(np.finfo(np.float32).eps)
            total[valid]+=np.arctanh(np.clip(data[valid],-1+epsilon,1-epsilon))
        expected=np.full(ref.shape,np.nan)
        valid=counts>0
        expected[valid]=np.tanh(total[valid]/counts[valid])
        saved=nib.load(str(root/'group'/('layer%02d_mean_r.nii.gz'%layer))).get_fdata()
        saved_n=nib.load(str(root/'group'/('layer%02d_n.nii.gz'%layer))).get_fdata()
        if not np.array_equal(saved_n,counts) or not np.array_equal(np.isfinite(saved),valid):
            raise ValueError('Group coverage mismatch')
        error=float(np.max(np.abs(expected[valid]-saved[valid])))
        if error>5e-8:raise ValueError('Group numerical mismatch')
        layer_reports.append({'layer':layer,'group_max_absolute_error':error,
                              'valid_voxels':int(valid.sum()),'all30_voxels':int((counts==30).sum()),
                              'r_min':float(saved[valid].min()),'r_max':float(saved[valid].max())})
        print('independent QC layer %d/12 complete'%layer,flush=True)
    for subject,valid in subject_valid.items():
        q=json.loads((root/'full'/subject/'qc.json').read_text())
        saved=nib.load(str(root/'full'/subject/'valid_coverage.nii.gz')).get_fdata()>0
        if not np.array_equal(saved,valid) or int(valid.sum())!=q['valid_centers']:
            raise ValueError('Subject coverage mismatch: '+subject)
        for item in q['inputs']:
            stat=Path(item['path']).stat()
            if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns']:
                raise ValueError('Input changed since run: '+item['path'])
    saved=nib.load(str(root/'group/common_all30_mask.nii.gz')).get_fdata()>0
    if not np.array_equal(saved,common):raise ValueError('Common mask mismatch')
    report={'passed':True,'subjects_checked':len(cfg['subjects']),'maps_checked':12*len(cfg['subjects']),
            'checks':['all map values and geometry','group estimates at every covered voxel',
                      'all coverage maps','input file size and modification time unchanged'],
            'common_all30_voxels':int(common.sum()),'layers':layer_reports}
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
