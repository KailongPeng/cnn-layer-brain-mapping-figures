"""Participant sign-flip max-|t| and subject bootstrap, conditional on fixed images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def t_from_sums(sums,sum_squares,n):
    variance_term=n*sum_squares-sums*sums
    if np.any(variance_term < -1e-8):raise ValueError('Invalid t variance')
    denominator=np.sqrt(np.maximum(variance_term,0))
    out=np.zeros(sums.shape,dtype=np.float64)
    np.divide(sums*np.sqrt(n-1),denominator,out=out,where=denominator>0)
    # Zero-variance columns must be handled explicitly before inference.
    if np.any((denominator==0)&(np.abs(sums)>1e-14)):
        raise ValueError('Nonzero constant column has undefined t statistic')
    return out


def max_t_pvalues(observed_t,null_max):
    ordered=np.sort(null_max)
    threshold=np.abs(observed_t)-1e-12*np.maximum(1,np.abs(observed_t))
    exceedances=len(ordered)-np.searchsorted(ordered,threshold,side='left')
    return (exceedances+1)/(len(ordered)+1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--config',required=True);p.add_argument('--inference-config',required=True)
    args=p.parse_args();root=Path(args.root);output=root/'inference'
    if output.exists():raise FileExistsError(str(output))
    cfg=json.loads(Path(args.config).read_text());settings_path=Path(args.inference_config)
    settings=json.loads(settings_path.read_text())
    subjects=cfg['subjects'];n=len(subjects)
    if n!=30 or settings['alternative']!='two-sided':raise ValueError('Unexpected inference specification')
    mask_path=root/'group/common_all30_mask.nii.gz';reference=nib.load(str(mask_path))
    mask=reference.get_fdata()>0;centers=int(mask.sum())
    z=np.empty((n,12,centers),dtype=np.float64)
    for i,subject in enumerate(subjects):
        for layer in range(12):
            img=nib.load(str(root/'full'/subject/('layer%02d_r.nii.gz'%(layer+1))))
            if img.shape!=mask.shape or not np.allclose(img.affine,reference.affine):raise ValueError('Inference space mismatch')
            data=img.get_fdata()[mask]
            if not np.isfinite(data).all() or np.any(np.abs(data)>1):raise ValueError('Invalid inference input')
            epsilon=np.finfo(np.float32).eps
            z[i,layer]=np.arctanh(np.clip(data,-1+epsilon,1-epsilon))
    flat=z.reshape(n,-1);squares=(flat*flat).sum(axis=0)
    observed=t_from_sums(flat.sum(axis=0),squares,n)
    rng=np.random.default_rng(settings['sign_flip_seed'])
    signs=rng.choice(np.array([-1.,1.]),size=(settings['n_sign_flips'],n))
    null_max=[];batch_size=settings['sign_flip_batch_size']
    for start in range(0,len(signs),batch_size):
        sums=signs[start:start+batch_size]@flat
        t=t_from_sums(sums,squares[None,:],n)
        null_max.extend(np.max(np.abs(t),axis=1).tolist())
        if start%512==0:print('sign flips %d/%d'%(start,len(signs)),flush=True)
    null_max=np.asarray(null_max)
    pvalues=max_t_pvalues(observed,null_max).reshape(12,centers)
    t_values=observed.reshape(12,centers)
    # Bootstrap participants jointly over all layers and locations. Stimuli fixed.
    rng=np.random.default_rng(settings['subject_bootstrap_seed'])
    samples=rng.integers(0,n,size=(settings['n_subject_bootstraps'],n))
    weights=np.asarray([np.bincount(row,minlength=n) for row in samples],dtype=np.float64)/n
    bootstrap=[]
    for start in range(0,len(weights),8):
        means=np.tanh((weights[start:start+8]@flat).reshape(-1,12,centers))
        means-=means.mean(axis=2,keepdims=True)
        covariance=means@means.transpose(0,2,1)
        norm=np.sqrt(np.diagonal(covariance,axis1=1,axis2=2))
        correlations=covariance/(norm[:,:,None]*norm[:,None,:])
        bootstrap.extend(correlations)
        if start%128==0:print('subject bootstraps %d/%d'%(start,len(weights)),flush=True)
    bootstrap=np.asarray(bootstrap)
    if not np.isfinite(bootstrap).all():raise ValueError('Invalid bootstrap correlations')
    low,high=np.quantile(bootstrap,[.025,.975],axis=0)
    point=np.loadtxt(str(root/'group/spatial_correlation_common_all30.csv'),delimiter=',')
    output.mkdir(parents=True,exist_ok=False)
    np.save(str(output/'null_max_abs_t.npy'),null_max,allow_pickle=False)
    np.save(str(output/'spatial_bootstrap_r.npy'),bootstrap,allow_pickle=False)
    np.savetxt(str(output/'spatial_pointwise95_low.csv'),low,delimiter=',')
    np.savetxt(str(output/'spatial_pointwise95_high.csv'),high,delimiter=',')
    for layer in range(12):
        for suffix,values,dtype in [('t',t_values[layer],np.float32),('p_fwe',pvalues[layer],np.float32),
                                    ('significant_fwe',pvalues[layer]<=settings['alpha'],np.uint8)]:
            data=np.full(mask.shape,0 if dtype==np.uint8 else np.nan,dtype=dtype)
            data[mask]=values
            nib.save(nib.Nifti1Image(data,reference.affine),str(output/('layer%02d_%s.nii.gz'%(layer+1,suffix))))
    with (output/'adjacent_layer_spatial_intervals.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['layer_a','layer_b','spatial_r','pointwise_95_low','pointwise_95_high'])
        for layer in range(11):writer.writerow([layer+1,layer+2,point[layer,layer+1],low[layer,layer+1],high[layer,layer+1]])
    report={'passed':True,'settings':settings,'settings_sha256':hashlib.sha256(settings_path.read_bytes()).hexdigest(),
            'mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest(),'n_subjects':n,'centers':centers,
            'n_tests':int(observed.size),'null_draws':len(null_max),
            'sign_patterns_sha256':hashlib.sha256(signs.astype(np.int8).tobytes()).hexdigest(),
            'significant_voxels_per_layer':(pvalues<=settings['alpha']).sum(axis=1).tolist(),
            'minimum_p':float(pvalues.min()),'group_r_maps_unchanged':True,
            'note':'max-|t| family covers all 12 layers jointly; bootstrap intervals are pointwise and fixed-stimulus'}
    (output/'qc.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
