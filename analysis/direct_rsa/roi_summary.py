"""Descriptive bilateral Wang ROI summaries using the existing paper atlas source.

Converts MATLAB one-based indices to NiBabel zero-based indices before resampling,
records the actual label names, and retains individual-subject ROI values.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.io import loadmat
from .atlas import fieldtrip_to_nibabel_affine


ROI_NAMES = {'V1':['v1v','v1d'], 'V2':['v2v','v2d'], 'V3':['v3v','v3d'],
             'hV4':['hv4'], 'VO':['vo1','vo2'], 'PHC':['phc1','phc2'],
             'LO':['lo1','lo2'], 'hMT':['hmt'], 'V3A/B':['v3a','v3b'],
             'IPS0':['ips0'], 'IPS1':['ips1'], 'IPS2':['ips2'], 'IPS3':['ips3']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--input-dir',required=True)
    p.add_argument('--group-dir',required=True)
    p.add_argument('--atlas-mat',required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    output=Path(args.output_dir)
    if output.exists():raise FileExistsError(str(output))
    group=Path(args.group_dir);source=Path(args.input_dir)
    reference=nib.load(str(group/'common_all30_mask.nii.gz'))
    common=reference.get_fdata()>0
    mat=loadmat(args.atlas_mat,squeeze_me=True,struct_as_record=False)['vtpm']
    labels=[str(x).strip() for x in mat.tissuelabel.tolist()]
    converted_affine=fieldtrip_to_nibabel_affine(mat.transform)
    atlas_source=nib.Nifti1Image(mat.tissue.astype(np.int16),converted_affine)
    atlas=resample_from_to(atlas_source,(reference.shape,reference.affine),order=0)
    tissue=atlas.get_fdata().astype(np.int16)
    masks=[];definitions=[]
    for name,suffixes in ROI_NAMES.items():
        ids=[i+1 for i,label in enumerate(labels) if label.lower().split('_')[-1] in suffixes]
        if len(ids)!=len(suffixes)*2:raise ValueError('Unexpected bilateral atlas labels: '+name)
        mask=np.isin(tissue,ids)&common
        if not mask.any():raise ValueError('Empty common-coverage ROI: '+name)
        masks.append(mask)
        definitions.append({'roi':name,'label_ids':ids,'label_names':[labels[i-1] for i in ids],
                            'atlas_voxels':int(np.isin(tissue,ids).sum()),'common_voxels':int(mask.sum())})
    v1=np.argwhere(masks[0]);world=nib.affines.apply_affine(reference.affine,v1)
    if np.mean(world[:,1])>-60 or not (np.min(world[:,0])<0<np.max(world[:,0])):
        raise ValueError('Bilateral posterior V1 geometry check failed')
    subject_values=np.empty((len(cfg['subjects']),12,len(masks)))
    group_values=np.empty((12,len(masks)))
    for layer in range(1,13):
        group_data=nib.load(str(group/('layer%02d_mean_r.nii.gz'%layer))).get_fdata()
        group_values[layer-1]=[np.mean(group_data[mask]) for mask in masks]
        for index,subject in enumerate(cfg['subjects']):
            img=nib.load(str(source/subject/('layer%02d_r.nii.gz'%layer)))
            if img.shape!=reference.shape or not np.allclose(img.affine,reference.affine):
                raise ValueError('ROI input space mismatch')
            data=img.get_fdata()
            subject_values[index,layer-1]=[np.mean(data[mask]) for mask in masks]
        print('ROI layer %d complete'%layer,flush=True)
    if not np.isfinite(subject_values).all():raise ValueError('Invalid values on common coverage')
    rng=np.random.default_rng(20261107)
    draws=rng.integers(0,len(cfg['subjects']),size=(1000,len(cfg['subjects'])))
    bootstrap=subject_values[draws].mean(axis=1)
    lo,hi=np.quantile(bootstrap,[.025,.975],axis=0)
    output.mkdir(parents=True,exist_ok=False)
    nib.save(atlas,str(output/'wang_atlas_mni2mm.nii.gz'))
    with (output/'subject_roi_r.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['subject','layer','roi','mean_voxel_r'])
        for index,subject in enumerate(cfg['subjects']):
            for layer in range(12):
                for roi,name in enumerate(ROI_NAMES):writer.writerow([subject,layer+1,name,subject_values[index,layer,roi]])
    with (output/'group_roi_r.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['layer','roi','mean_subject_roi_r','bootstrap_95_low','bootstrap_95_high','mean_group_fisher_map_r'])
        for layer in range(12):
            for roi,name in enumerate(ROI_NAMES):
                writer.writerow([layer+1,name,subject_values[:,layer,roi].mean(),lo[layer,roi],hi[layer,roi],group_values[layer,roi]])
    report={'descriptive_only':True,'definitions':definitions,'atlas_mat':args.atlas_mat,
            'atlas_mat_sha256':hashlib.sha256(Path(args.atlas_mat).read_bytes()).hexdigest(),
            'conversion':'FieldTrip one-based transform @ translation(+1,+1,+1) for NiBabel zero-based voxel indices; nearest-neighbor resampling',
            'source_affine':np.asarray(mat.transform).tolist(),'nibabel_affine':converted_affine.tolist(),
            'coordinate_source':'https://www.fieldtriptoolbox.org/faq/homogenous/',
            'roi_statistic':'within-ROI arithmetic mean of signed r per subject; arithmetic mean across subjects',
            'additional_statistic':'spatial ROI mean of group Fisher-aggregated map; separate CSV column',
            'uncertainty':'1000 subject bootstrap resamples; pointwise percentile intervals; no multiplicity correction',
            'seed':20261107,'n_subjects':len(cfg['subjects']),
            'v1_world_centroid_mm':world.mean(axis=0).tolist()}
    (output/'qc.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
