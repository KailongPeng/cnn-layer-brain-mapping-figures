"""Independent full-output checks for NSD, image splits, and participant inference."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import pearsonr,ttest_1samp


def check_nsd(root):
    folder=root/'nsd';manifest=json.loads((folder/'manifest.json').read_text())
    group=folder/'group';reference=nib.load(str(group/'common_all8_all16_mask.nii.gz'))
    common=reference.get_fdata()>0;all_finite=np.ones(common.shape,dtype=bool)
    coords=np.argwhere(common)
    sample=coords[np.random.default_rng(802).choice(len(coords),32,replace=False)]
    expected=np.empty((8,16,12,32))
    for index,job in enumerate(manifest['jobs']):
        q=json.loads((folder/'full'/('job%d'%index)/'qc.json').read_text())
        if hashlib.sha256(Path(job['model_rsms']).read_bytes()).hexdigest()!=job['model_sha256']:
            raise ValueError('NSD model changed after execution')
        subject=int(job['subject'].split('sub_i')[1])-1;which_set=job['which_set']-1
        coverage=None
        for layer in range(12):
            img=nib.load(str(folder/'full'/('job%d'%index)/('layer%02d_r.nii.gz'%(layer+1))))
            if img.shape!=common.shape or not np.allclose(img.affine,reference.affine):raise ValueError('NSD map space mismatch')
            data=img.get_fdata();valid=np.isfinite(data)
            if np.isinf(data).any() or np.any(np.abs(data[valid])>1):raise ValueError('NSD invalid Pearson value')
            coverage=valid if coverage is None else coverage&valid
            all_finite&=valid
            expected[subject,which_set,layer]=data[tuple(sample.T)]
        saved=nib.load(str(folder/'full'/('job%d'%index)/'valid_coverage.nii.gz')).get_fdata()>0
        if not np.array_equal(coverage,saved) or int(coverage.sum())!=q['valid_centers']:
            raise ValueError('NSD subject-set coverage mismatch')
        for item in q['inputs']:
            stat=Path(item['path']).stat()
            if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns']:
                raise ValueError('NSD input changed after execution')
        print('NSD independent QC %d/128'%(index+1),flush=True)
    if not np.array_equal(common,all_finite):raise ValueError('NSD group common coverage mismatch')
    epsilon=np.finfo(np.float32).eps
    expected=np.tanh(np.arctanh(np.clip(expected,-1+epsilon,1-epsilon)).mean(axis=1).mean(axis=0))
    actual=np.asarray([nib.load(str(group/('layer%02d_mean_r.nii.gz'%layer))).get_fdata()[tuple(sample.T)] for layer in range(1,13)])
    if not np.allclose(expected,actual,atol=1e-7,rtol=1e-7):raise ValueError('NSD hierarchical group estimate mismatch')
    return {'passed':True,'maps_checked':1536,'subject_set_coverages_checked':128,
            'common_voxels':int(common.sum()),'group_sample_layer_points':int(expected.size),
            'group_sample_max_error':float(np.max(np.abs(expected-actual)))}


def check_splits(root,cfg):
    folder=root/'image_splits';plan_path=folder/'plan.json';plan=json.loads(plan_path.read_text())
    plan_sha=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    common=nib.load(str(root/'group/common_all30_mask.nii.gz')).get_fdata()>0
    coordinates=np.argwhere(common);matrices=[];largest_error=0
    for batch in sorted(folder.glob('batch*')):
        q=json.loads((batch/'qc.json').read_text())
        if not q['passed'] or q['plan_sha256']!=plan_sha:raise ValueError('Split batch provenance mismatch')
        if not np.array_equal(np.load(str(batch/'coordinates.npy')),coordinates):raise ValueError('Split coordinate order mismatch')
        for split in sorted(batch.glob('split*')):
            q=json.loads((split/'qc.json').read_text());index=q['split']['id']
            if q['split']!=plan['splits'][index]:raise ValueError('Split image indices mismatch')
            maps=np.load(str(split/'group_r.npy'),allow_pickle=False)
            if maps.shape!=(2,len(coordinates),12) or not np.isfinite(maps).all() or np.any(np.abs(maps)>1):
                raise ValueError('Invalid split group maps')
            a=maps[0].astype(np.float64);b=maps[1].astype(np.float64)
            a-=a.mean(axis=0);b-=b.mean(axis=0)
            matrix=(a.T@b)/(np.linalg.norm(a,axis=0)[:,None]*np.linalg.norm(b,axis=0)[None,:])
            saved=np.loadtxt(str(split/'cross_half_spatial_r.csv'),delimiter=',')
            error=float(np.max(np.abs(matrix-saved)));largest_error=max(error,largest_error)
            if error>1e-10:raise ValueError('Split spatial matrix differs from saved maps')
            matrices.append((index,saved))
    if sorted(index for index,matrix in matrices)!=list(range(100)):raise ValueError('Split output count mismatch')
    # Independently rebuild two splits from raw cope patterns at 16 fixed centers.
    chosen=np.random.default_rng(803).choice(len(coordinates),16,replace=False)
    coords=coordinates[chosen];models=np.load(cfg['model_rsms']);split_ids=[0,99]
    sums=np.zeros((2,2,16,12));epsilon=np.finfo(np.float32).eps
    for subject in cfg['subjects']:
        per_image=[]
        for i in range(1,17):
            data=nib.load(cfg['cope'].format(subject=subject,image=i)).get_fdata()
            per_image.append(np.asarray([data[tuple(slice(int(k)-3,int(k)+4) for k in coord)].reshape(-1) for coord in coords]))
        patterns=np.stack(per_image,axis=1)
        for si,index in enumerate(split_ids):
            for hi,key in enumerate(['a','b']):
                ix=plan['splits'][index][key];tri=np.triu_indices(8,1)
                model=models[:,np.asarray(ix)[:,None],np.asarray(ix)[None,:]][:,tri[0],tri[1]]
                for point in range(len(coords)):
                    brain=np.corrcoef(patterns[point,ix])[tri]
                    r=np.asarray([pearsonr(x,brain)[0] for x in model])
                    sums[si,hi,point]+=np.arctanh(np.clip(r,-1+epsilon,1-epsilon))
        print('raw split independent QC subject '+subject,flush=True)
    raw_error=0
    for si,index in enumerate(split_ids):
        paths=list(folder.glob('batch*/split%03d/group_r.npy'%index))
        if len(paths)!=1:raise ValueError('Missing raw split check output')
        saved=np.load(str(paths[0]),allow_pickle=False)[:,chosen,:]
        expected=np.tanh(sums[si]/30)
        raw_error=max(raw_error,float(np.max(np.abs(saved-expected))))
        if not np.allclose(saved,expected,atol=1e-7,rtol=1e-7):raise ValueError('Raw-pattern split reconstruction mismatch')
    stacked=np.asarray([matrix for index,matrix in sorted(matrices)])
    saved=np.load(str(folder/'summary/all_cross_half_spatial_r.npy'),allow_pickle=False)
    if not np.allclose(stacked,saved,atol=1e-12):raise ValueError('Split summary array mismatch')
    return {'passed':True,'split_maps_checked':100,'spatial_matrices_checked':100,
            'max_spatial_matrix_error':largest_error,'raw_pattern_splits_checked':split_ids,
            'raw_pattern_layer_center_comparisons':int(sums.size),'raw_pattern_max_group_error':raw_error,
            'quantiles_are_not_confidence_intervals':True}


def check_inference(root,cfg):
    folder=root/'inference';report=json.loads((folder/'qc.json').read_text())
    settings=report['settings'];subjects=cfg['subjects'];n=len(subjects)
    common=nib.load(str(root/'group/common_all30_mask.nii.gz')).get_fdata()>0
    z=[];epsilon=np.finfo(np.float32).eps
    for subject in subjects:
        z.append([np.arctanh(np.clip(nib.load(str(root/'full'/subject/('layer%02d_r.nii.gz'%layer))).get_fdata()[common],-1+epsilon,1-epsilon)) for layer in range(1,13)])
    z=np.asarray(z);expected_t=ttest_1samp(z,0,axis=0).statistic
    null=np.load(str(folder/'null_max_abs_t.npy'),allow_pickle=False)
    signs=np.random.default_rng(settings['sign_flip_seed']).choice(np.array([-1.,1.]),size=(settings['n_sign_flips'],n))
    null_errors=[]
    for i in [0,1]:
        expected=float(np.max(np.abs(ttest_1samp(z*signs[i,:,None,None],0,axis=0).statistic)))
        null_errors.append(abs(expected-null[i]))
        if not np.isclose(expected,null[i],atol=1e-9,rtol=1e-9):raise ValueError('Inference null maximum mismatch')
    t_error=0;significant=[]
    for layer in range(12):
        t=nib.load(str(folder/('layer%02d_t.nii.gz'%(layer+1)))).get_fdata()[common]
        p=nib.load(str(folder/('layer%02d_p_fwe.nii.gz'%(layer+1)))).get_fdata()[common]
        threshold=np.abs(expected_t[layer])-1e-12*np.maximum(1,np.abs(expected_t[layer]))
        expected_p=(1+len(null)-np.searchsorted(np.sort(null),threshold,side='left'))/(len(null)+1)
        if not np.allclose(t,expected_t[layer],atol=1e-6,rtol=1e-6):raise ValueError('Observed t differs from SciPy')
        if not np.allclose(p,expected_p,atol=1e-7,rtol=1e-7):raise ValueError('Max-T p map mismatch')
        mask=nib.load(str(folder/('layer%02d_significant_fwe.nii.gz'%(layer+1)))).get_fdata()[common]>0
        if not np.array_equal(mask,expected_p<=settings['alpha']):raise ValueError('Inference significance mask mismatch')
        significant.append(int(mask.sum()));t_error=max(t_error,float(np.max(np.abs(t-expected_t[layer]))))
    bootstrap=np.load(str(folder/'spatial_bootstrap_r.npy'),allow_pickle=False)
    if bootstrap.shape!=(1000,12,12) or not np.isfinite(bootstrap).all():raise ValueError('Invalid spatial bootstrap output')
    if not np.allclose(bootstrap,bootstrap.transpose(0,2,1)) or not np.allclose(np.diagonal(bootstrap,axis1=1,axis2=2),1):raise ValueError('Invalid bootstrap correlation matrices')
    low,high=np.quantile(bootstrap,[.025,.975],axis=0)
    if not np.allclose(low,np.loadtxt(str(folder/'spatial_pointwise95_low.csv'),delimiter=',')) or not np.allclose(high,np.loadtxt(str(folder/'spatial_pointwise95_high.csv'),delimiter=',')):
        raise ValueError('Spatial percentile interval mismatch')
    return {'passed':True,'n_tests':int(expected_t.size),'null_draws':len(null),
            'max_saved_t_error':t_error,'independently_checked_null_indices':[0,1],
            'max_null_maximum_error':max(null_errors),'significant_voxels_per_layer':significant,
            'bootstrap_matrices_checked':len(bootstrap)}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['nsd','splits','inference'])
    p.add_argument('--root',required=True);p.add_argument('--config',required=True)
    args=p.parse_args();root=Path(args.root);cfg=json.loads(Path(args.config).read_text())
    folders={'nsd':'nsd','splits':'image_splits','inference':'inference'}
    output=root/folders[args.mode]/'independent_qc.json'
    if output.exists():raise FileExistsError(str(output))
    if args.mode=='nsd':report=check_nsd(root)
    elif args.mode=='splits':report=check_splits(root,cfg)
    else:report=check_inference(root,cfg)
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
