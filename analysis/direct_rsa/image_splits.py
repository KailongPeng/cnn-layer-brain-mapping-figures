"""Repeated complementary image splits; split distributions are descriptive, not iid tests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .cached_rsa import score_cached
from .core import paired_image_halves


def prepare(root,config):
    output=root/'image_splits/plan.json'
    if output.exists():raise FileExistsError(str(output))
    mask=root/'group/common_all30_mask.nii.gz'
    splits=paired_image_halves(n_splits=100,seed=20261107)
    plan={'seed':20261107,'n_splits':100,'subjects':config['subjects'],
          'model_rsms':config['model_rsms'],
          'model_sha256':hashlib.sha256(Path(config['model_rsms']).read_bytes()).hexdigest(),
          'mask_sha256':hashlib.sha256(mask.read_bytes()).hexdigest(),
          'splits':[{'id':i,'a':a.tolist(),'b':b.tolist()} for i,(a,b) in enumerate(splits)],
          'n_unique_unordered_partitions':len({tuple(min(tuple(a),tuple(b))) for a,b in splits}),
          'aggregation':'within each half: tanh(mean(subject atanh(r))); equal weight for all 30 subjects',
          'spatial_comparison':'same fixed common mask for both halves, all layers, all splits',
          'interpretation':'overlapping image/subject samples; split quantiles are not confidence intervals'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(plan,indent=2,allow_nan=False))
    print(json.dumps({'plan':str(output),'n_splits':100,'unique':plan['n_unique_unordered_partitions']}))


def cross_half_spatial(a,b):
    """a,b: centers x layers. Return A-layer by B-layer spatial Pearson."""
    if a.shape!=b.shape or a.ndim!=2 or len(a)<3:raise ValueError('Invalid split maps')
    return np.corrcoef(a.T,b.T)[:a.shape[1],a.shape[1]:]


def run_batch(root,batch_index,batch_size):
    plan_path=root/'image_splits/plan.json';plan=json.loads(plan_path.read_text())
    batch=plan['splits'][batch_index*batch_size:(batch_index+1)*batch_size]
    if not batch:raise ValueError('Empty split batch')
    output=root/'image_splits'/('batch%02d'%batch_index)
    if output.exists():raise FileExistsError(str(output))
    if len(plan['subjects'])!=30:raise ValueError('Expected 30 subjects')
    if hashlib.sha256(Path(plan['model_rsms']).read_bytes()).hexdigest()!=plan['model_sha256']:
        raise ValueError('Split model changed')
    models=np.load(plan['model_rsms'],allow_pickle=False)
    sums=None;coords=None;commits=set()
    for subject in plan['subjects']:
        folder=root/'brain_rsm_cache'/subject
        qc=json.loads((folder/'qc.json').read_text())
        if not qc['passed'] or qc['mask_sha256']!=plan['mask_sha256'] or qc['model_sha256']!=plan['model_sha256']:
            raise ValueError('Cache provenance mismatch: '+subject)
        commits.add(qc['code_commit'])
        current=np.load(str(folder/'coordinates.npy'),allow_pickle=False)
        cache=np.load(str(folder/'brain_upper.npy'),allow_pickle=False,mmap_mode='r')
        if coords is None:
            coords=current;sums=np.zeros((len(batch),2,len(coords),12),dtype=np.float64)
        if not np.array_equal(coords,current):raise ValueError('Cache center order differs')
        for i,split in enumerate(batch):
            for half,key in enumerate(['a','b']):
                values=score_cached(models,cache,split[key])
                if not np.isfinite(values).all():raise ValueError('Undefined subset r; inspect before changing coverage')
                epsilon=np.finfo(np.float32).eps
                sums[i,half]+=np.arctanh(np.clip(values,-1+epsilon,1-epsilon))
        del cache
        print('batch %d subject %s complete'%(batch_index,subject),flush=True)
    maps=np.tanh(sums/len(plan['subjects'])).astype(np.float32)
    output.mkdir(parents=True,exist_ok=False)
    np.save(str(output/'coordinates.npy'),coords,allow_pickle=False)
    for i,split in enumerate(batch):
        folder=output/('split%03d'%split['id']);folder.mkdir(exist_ok=False)
        matrix=cross_half_spatial(maps[i,0],maps[i,1])
        if not np.isfinite(matrix).all():raise ValueError('Undefined split spatial correlation')
        diagonal=np.diag(matrix)
        different=(matrix.sum(0)+matrix.sum(1)-2*diagonal)/22
        np.save(str(folder/'group_r.npy'),maps[i],allow_pickle=False)
        np.savetxt(str(folder/'cross_half_spatial_r.csv'),matrix,delimiter=',')
        with (folder/'layer_summary.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['layer','same_layer_r','mean_different_layer_r','same_minus_different'])
            for layer in range(12):writer.writerow([layer+1,diagonal[layer],different[layer],diagonal[layer]-different[layer]])
        meta={'split':split,'n_subjects':30,'centers':len(coords),'group_map_shape':list(maps[i].shape),
              'map_axes':['half_A_B','coordinate_row','layer'],'descriptive_only':True}
        (folder/'qc.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    (output/'qc.json').write_text(json.dumps({'passed':True,'split_ids':[x['id'] for x in batch],
         'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'cache_code_commits':sorted(commits)},indent=2))


def summarize(root):
    plan=json.loads((root/'image_splits/plan.json').read_text())
    output=root/'image_splits/summary'
    if output.exists():raise FileExistsError(str(output))
    matrices=[];ids=[];differences=[]
    for folder in sorted((root/'image_splits').glob('batch*/split*')):
        qc=json.loads((folder/'qc.json').read_text());matrix=np.loadtxt(str(folder/'cross_half_spatial_r.csv'),delimiter=',')
        matrices.append(matrix);ids.append(qc['split']['id'])
        diagonal=np.diag(matrix);different=(matrix.sum(0)+matrix.sum(1)-2*diagonal)/22
        differences.append(np.stack([diagonal,different,diagonal-different],axis=1))
    if sorted(ids)!=list(range(plan['n_splits'])):raise ValueError('Incomplete or duplicate image split outputs')
    matrices=np.asarray(matrices);values=np.asarray(differences)
    output.mkdir(parents=True,exist_ok=False)
    np.savetxt(str(output/'mean_cross_half_spatial_r.csv'),matrices.mean(0),delimiter=',')
    np.save(str(output/'all_cross_half_spatial_r.npy'),matrices,allow_pickle=False)
    with (output/'layer_distribution.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['layer','measure','mean','median','q025','q25','q75','q975'])
        for layer in range(12):
            for measure,name in enumerate(['same_layer_r','mean_different_layer_r','same_minus_different']):
                sample=values[:,layer,measure];q=np.quantile(sample,[.5,.025,.25,.75,.975])
                writer.writerow([layer+1,name,float(sample.mean())]+q.tolist())
    (output/'qc.json').write_text(json.dumps({'passed':True,'n_splits':len(ids),'n_subjects':30,
        'descriptive_only':True,'quantiles_are_not_confidence_intervals':True,
        'generalization':'robustness to subsets of the existing 16 images, not new independently synthesized image sets'},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','batch','summarize'])
    p.add_argument('--root',required=True);p.add_argument('--config')
    p.add_argument('--batch-index',type=int,default=0);p.add_argument('--batch-size',type=int,default=10)
    args=p.parse_args();root=Path(args.root)
    if args.mode=='prepare':prepare(root,json.loads(Path(args.config).read_text()))
    elif args.mode=='batch':run_batch(root,args.batch_index,args.batch_size)
    else:summarize(root)


if __name__=='__main__':main()
