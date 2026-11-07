"""Hierarchical descriptive NSD aggregation: sets within subject, then subjects."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root);source=root/'nsd';output=source/'group'
    if output.exists():raise FileExistsError(str(output))
    path=source/'manifest.json';manifest=json.loads(path.read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if len(manifest['jobs'])!=128:raise ValueError('Incomplete NSD manifest')
    for index,job in enumerate(manifest['jobs']):
        qc=json.loads((source/'full'/('job%d'%index)/'qc.json').read_text())
        if (qc['validation_only'] or qc['subject']!=job['subject'] or qc['which_set']!=job['which_set'] or
            qc['manifest_sha256']!=digest or qc['model_sha256']!=job['model_sha256']):
            raise ValueError('NSD full output provenance mismatch')
    reference=nib.load(manifest['jobs'][0]['mask']);shape=reference.shape
    output.mkdir(parents=True,exist_ok=False)
    common=np.ones(shape,dtype=bool);layer_qc=[]
    epsilon=np.finfo(np.float32).eps
    for layer in range(1,13):
        subject_sum=np.zeros((8,)+shape);subject_n=np.zeros((8,)+shape,dtype=np.uint8)
        set_sum=np.zeros((16,)+shape);set_n=np.zeros((16,)+shape,dtype=np.uint8)
        for index,job in enumerate(manifest['jobs']):
            image=nib.load(str(source/'full'/('job%d'%index)/('layer%02d_r.nii.gz'%layer)))
            if image.shape!=shape or not np.allclose(image.affine,reference.affine,atol=1e-4):raise ValueError('NSD map geometry mismatch')
            values=image.get_fdata();valid=np.isfinite(values)
            if np.isinf(values).any() or np.any(np.abs(values[valid])>1):raise ValueError('Invalid NSD Pearson range')
            z=np.zeros(shape);z[valid]=np.arctanh(np.clip(values[valid],-1+epsilon,1-epsilon))
            subject=int(job['subject'].split('sub_i')[1])-1;which_set=job['which_set']-1
            subject_sum[subject]+=z;subject_n[subject]+=valid
            set_sum[which_set]+=z;set_n[which_set]+=valid
        complete=subject_n==16
        subject_z=subject_sum/16
        for subject in range(8):
            folder=output/('sub_i%d'%(subject+1));folder.mkdir(exist_ok=True)
            mean=np.full(shape,np.nan,dtype=np.float32);mean[complete[subject]]=np.tanh(subject_z[subject][complete[subject]])
            nib.save(nib.Nifti1Image(mean,reference.affine),str(folder/('layer%02d_mean16_r.nii.gz'%layer)))
        n=complete.sum(axis=0)
        sum_z=np.where(complete,subject_z,0).sum(axis=0)
        group=np.full(shape,np.nan,dtype=np.float32)
        group[n>0]=np.tanh(sum_z[n>0]/n[n>0])
        nib.save(nib.Nifti1Image(group,reference.affine),str(output/('layer%02d_mean_r.nii.gz'%layer)))
        nib.save(nib.Nifti1Image(n.astype(np.uint8),reference.affine),str(output/('layer%02d_n_complete16.nii.gz'%layer)))
        common&=n==8
        for which_set in range(16):
            folder=output/('set%02d'%(which_set+1));folder.mkdir(exist_ok=True)
            values=np.full(shape,np.nan,dtype=np.float32);valid=set_n[which_set]>0
            values[valid]=np.tanh(set_sum[which_set][valid]/set_n[which_set][valid])
            nib.save(nib.Nifti1Image(values,reference.affine),str(folder/('layer%02d_mean8_r.nii.gz'%layer)))
            nib.save(nib.Nifti1Image(set_n[which_set],reference.affine),str(folder/('layer%02d_n.nii.gz'%layer)))
        layer_qc.append({'layer':layer,'all8_complete16_voxels':int((n==8).sum()),'any_coverage_voxels':int((n>0).sum())})
        print('NSD group layer %d complete'%layer,flush=True)
    if common.sum()<3:raise ValueError('Insufficient NSD common coverage')
    nib.save(nib.Nifti1Image(common.astype(np.uint8),reference.affine),str(output/'common_all8_all16_mask.nii.gz'))
    maps=np.asarray([nib.load(str(output/('layer%02d_mean_r.nii.gz'%layer))).get_fdata()[common] for layer in range(1,13)])
    np.savetxt(str(output/'spatial_correlation_common_all8_all16.csv'),np.corrcoef(maps),delimiter=',')
    report={'passed':True,'descriptive_only':True,'n_subjects':8,'n_sets':16,'n_input_maps':1536,
            'manifest_sha256':digest,'common_all8_all16_voxels':int(common.sum()),'layers':layer_qc,
            'aggregation':'mean Fisher z across 16 sets within each subject; equal-weight mean across subjects; inverse tanh',
            'coverage':'subject estimate requires all 16 sets; primary comparison mask requires all 8 subjects at all 12 layers',
            'limitations':'different participants, repetitions and image sets from synthesized experiment; raw r differences are not isolated stimulus-method effects'}
    (output/'qc.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
