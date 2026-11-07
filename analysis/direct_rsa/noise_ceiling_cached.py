"""Noise-ceiling context using exactly the brain RSMs used for new direct r maps."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def ceiling_correlations(vectors):
    """subjects x centers x pairs -> lower and upper subject correlations."""
    x=np.asarray(vectors,dtype=np.float64)
    if x.ndim!=3 or len(x)<3 or not np.isfinite(x).all():raise ValueError('Invalid noise-ceiling input')
    x=x-x.mean(axis=2,keepdims=True)
    total=x.sum(axis=0)
    others=total[None,:]-x
    norm=np.linalg.norm(x,axis=2)
    with np.errstate(invalid='ignore',divide='ignore'):
        lower=np.sum(x*others,axis=2)/(norm*np.linalg.norm(others,axis=2))
        upper=np.sum(x*total[None,:],axis=2)/(norm*np.linalg.norm(total,axis=1)[None,:])
    return lower,upper


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True)
    args=p.parse_args();root=Path(args.root);cfg=json.loads(Path(args.config).read_text())
    output=root/'noise_ceiling'
    if output.exists():raise FileExistsError(str(output))
    mask_img=nib.load(str(root/'group/common_all30_mask.nii.gz'));mask=mask_img.get_fdata()>0
    coords=np.argwhere(mask);caches=[]
    for subject in cfg['subjects']:
        folder=root/'brain_rsm_cache'/subject;q=json.loads((folder/'qc.json').read_text())
        if not q['passed'] or not np.array_equal(np.load(str(folder/'coordinates.npy')),coords):raise ValueError('Noise-ceiling cache mismatch')
        caches.append(np.load(str(folder/'brain_upper.npy'),allow_pickle=False,mmap_mode='r'))
    lower=np.empty((30,len(coords)),dtype=np.float32);upper=np.empty_like(lower)
    for start in range(0,len(coords),2048):
        stop=min(start+2048,len(coords));chunk=np.asarray([x[start:stop] for x in caches])
        lo,hi=ceiling_correlations(chunk)
        lower[:,start:stop]=lo;upper[:,start:stop]=hi
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():raise ValueError('Undefined noise-ceiling correlation')
    output.mkdir(parents=True,exist_ok=False)
    np.save(str(output/'per_subject_lower_r.npy'),lower,allow_pickle=False)
    np.save(str(output/'per_subject_upper_r.npy'),upper,allow_pickle=False)
    for name,values in [('lower',lower),('upper',upper)]:
        data=np.full(mask.shape,np.nan,dtype=np.float32);data[mask]=values.mean(axis=0,dtype=np.float64)
        nib.save(nib.Nifti1Image(data,mask_img.affine),str(output/(name+'_mean_r.nii.gz')))
    atlas=nib.load(str(root/'roi_corrected/wang_atlas_mni2mm.nii.gz')).get_fdata()
    definitions=json.loads((root/'roi_corrected/qc.json').read_text())['definitions']
    model_rows=list(csv.DictReader((root/'roi_corrected/group_roi_r.csv').open()))
    roi_rows=[]
    fig,axes=plt.subplots(4,4,figsize=(15,11),sharex=True,constrained_layout=True)
    for index,definition in enumerate(definitions):
        selected=np.isin(atlas[mask],definition['label_ids'])
        lo=float(lower[:,selected].mean(dtype=np.float64));hi=float(upper[:,selected].mean(dtype=np.float64))
        rows=sorted([row for row in model_rows if row['roi']==definition['roi']],key=lambda row:int(row['layer']))
        values=np.asarray([float(row['mean_subject_roi_r']) for row in rows])
        ax=axes.flat[index];ax.axhspan(lo,hi,color='gray',alpha=.22,label='LOSO-to-inclusive RSM reliability')
        ax.plot(range(1,13),values,marker='o',markersize=3,color='#126ba0')
        ax.axhline(0,color='black',linewidth=.6);ax.set_title(definition['roi']);ax.set_xticks([1,4,8,12]);ax.tick_params(axis='x',labelbottom=True)
        ax.set_xlabel('Layer');ax.set_ylabel('Mean raw Pearson r')
        roi_rows.append({'roi':definition['roi'],'common_voxels':int(selected.sum()),'lower_mean_r':lo,'upper_mean_r':hi})
    for index in range(len(definitions),16):axes.flat[index].set_visible(False)
    fig.suptitle('Current 16-image synthesized data: model r and brain-RSM reliability\n'
                 'Same full-Cube input and common coverage; gray band is not a confidence interval')
    fig.savefig(str(output/'roi_model_r_and_noise_ceiling.png'),dpi=140)
    fig.savefig(str(output/'roi_model_r_and_noise_ceiling.pdf'));plt.close(fig)
    with (output/'roi_noise_ceiling.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['roi','common_voxels','lower_mean_r','upper_mean_r']);writer.writeheader();writer.writerows(roi_rows)
    report={'passed':True,'n_subjects':30,'centers':len(coords),'visually_reviewed':False,
            'method':'lower: corr(subject brain RSM, mean other-subject RSM); upper: corr(subject RSM, inclusive mean RSM)',
            'aggregation':'arithmetic mean raw r across subjects, then across ROI voxels; matches ROI model-r averaging',
            'roi_source':'roi_corrected with MATLAB one-based origin corrected',
            'input':'exact float64 full-Cube brain RSM cache used for direct r; no removal of constant-across-image voxel columns',
            'interpretation':'finite-sample lower/upper estimates of RSM reliability; band is not an inferential confidence interval',
            'historical_difference':'old noise-ceiling worker removed voxel columns with zero across-image variance; do not assume identical old/new estimands'}
    (output/'qc.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
