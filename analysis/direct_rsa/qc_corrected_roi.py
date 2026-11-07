"""Check atlas labels by direct world-to-MATLAB lookup and verify all ROI values."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from scipy.io import loadmat


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True)
    p.add_argument('--background',required=True);args=p.parse_args()
    root=Path(args.root);roi=root/'roi_corrected';output=roi/'independent_qc.json'
    if output.exists():raise FileExistsError(str(output))
    cfg=json.loads(Path(args.config).read_text());q=json.loads((roi/'qc.json').read_text())
    atlas_img=nib.load(str(roi/'wang_atlas_mni2mm.nii.gz'));atlas=atlas_img.get_fdata().astype(int)
    ref=nib.load(str(root/'group/common_all30_mask.nii.gz'));common=ref.get_fdata()>0
    if atlas.shape!=common.shape or not np.allclose(atlas_img.affine,ref.affine):raise ValueError('Corrected atlas geometry mismatch')
    mat=loadmat(q['atlas_mat'],squeeze_me=True,struct_as_record=False)['vtpm']
    coordinates=np.indices(atlas.shape).reshape(3,-1).T
    world=nib.affines.apply_affine(atlas_img.affine,coordinates)
    matlab=nib.affines.apply_affine(np.linalg.inv(mat.transform),world)
    zero_based=matlab-1
    nearest=np.floor(zero_based+.5).astype(int)
    valid=np.all((zero_based>=0)&(zero_based<=np.asarray(mat.tissue.shape)-1),axis=1)
    expected=np.zeros(len(coordinates),dtype=int)
    expected[valid]=mat.tissue[tuple(nearest[valid].T)]
    mismatch=int(np.count_nonzero(expected.reshape(atlas.shape)!=atlas))
    if mismatch:raise ValueError('Atlas labels disagree with direct MATLAB coordinate lookup')
    definitions=q['definitions'];masks=[np.isin(atlas,d['label_ids'])&common for d in definitions]
    values=np.empty((30,12,len(definitions)))
    raw=list(csv.DictReader((roi/'subject_roi_r.csv').open()))
    indexed={(r['subject'],int(r['layer']),r['roi']):float(r['mean_voxel_r']) for r in raw}
    maximum=0
    for si,subject in enumerate(cfg['subjects']):
        for layer in range(1,13):
            data=nib.load(str(root/'full'/subject/('layer%02d_r.nii.gz'%layer))).get_fdata()
            means=np.asarray([np.mean(data[mask]) for mask in masks]);values[si,layer-1]=means
            saved=np.asarray([indexed[(subject,layer,d['roi'])] for d in definitions])
            maximum=max(maximum,float(np.max(np.abs(means-saved))))
    if maximum>1e-12:raise ValueError('Corrected ROI subject CSV mismatch')
    rows=list(csv.DictReader((roi/'group_roi_r.csv').open()))
    indexed={(int(r['layer']),r['roi']):r for r in rows}
    draws=np.random.default_rng(20261107).integers(0,30,size=(1000,30))
    lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975],axis=0)
    for layer in range(12):
        data=nib.load(str(root/'group'/('layer%02d_mean_r.nii.gz'%(layer+1)))).get_fdata()
        for ri,d in enumerate(definitions):
            row=indexed[(layer+1,d['roi'])]
            a=[values[:,layer,ri].mean(),lo[layer,ri],hi[layer,ri],np.mean(data[masks[ri]])]
            b=[float(row[key]) for key in ['mean_subject_roi_r','bootstrap_95_low','bootstrap_95_high','mean_group_fisher_map_r']]
            if not np.allclose(a,b,atol=1e-12,rtol=1e-12):raise ValueError('Corrected ROI group CSV mismatch')
    background=nib.load(args.background)
    if background.shape!=atlas.shape or not np.allclose(background.affine,atlas_img.affine):raise ValueError('Atlas background space mismatch')
    canonical=nib.as_closest_canonical(background);brain=canonical.get_fdata()
    canonical_atlas=nib.as_closest_canonical(atlas_img).get_fdata();affine=canonical.affine
    selected=[d for d in definitions if d['roi'] in ['V1','V2','V3','hV4','LO','PHC']]
    colors=['#e41a1c','#ff7f00','#4daf4a','#984ea3','#377eb8','#ffff33']
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    vmax=float(np.percentile(brain[brain>0],99))
    for column,z in enumerate([-8,8,24]):
        k=int(round(nib.affines.apply_affine(np.linalg.inv(affine),[0,0,z])[2]))
        ax=axes[0,column];ax.imshow(brain[:,:,k].T,origin='lower',cmap='gray',vmin=0,vmax=vmax)
        for d,color in zip(selected,colors):
            mask=np.isin(canonical_atlas[:,:,k],d['label_ids'])
            if mask.any():ax.contour(mask.T,levels=[.5],colors=[color],linewidths=.7)
        ax.set_title('Axial z=%d mm'%z);ax.set_xticks([]);ax.set_yticks([])
    for column,y in enumerate([-96,-80,-64]):
        j=int(round(nib.affines.apply_affine(np.linalg.inv(affine),[0,y,0])[1]))
        ax=axes[1,column];ax.imshow(brain[:,j,:].T,origin='lower',cmap='gray',vmin=0,vmax=vmax)
        for d,color in zip(selected,colors):
            mask=np.isin(canonical_atlas[:,j,:],d['label_ids'])
            if mask.any():ax.contour(mask.T,levels=[.5],colors=[color],linewidths=.7)
        ax.set_title('Coronal y=%d mm'%y);ax.set_xticks([]);ax.set_yticks([])
    fig.legend(handles=[Line2D([0],[0],color=color,label=d['roi']) for d,color in zip(selected,colors)],loc='lower center',ncol=6)
    fig.suptitle('Corrected FieldTrip-to-NIfTI atlas on FEAT MNI152 template\nNeurological orientation; MATLAB one-based voxel origin accounted for')
    fig.savefig(str(roi/'atlas_alignment.png'),dpi=150);fig.savefig(str(roi/'atlas_alignment.pdf'));plt.close(fig)
    report={'passed':True,'atlas_voxels_checked':int(atlas.size),'direct_coordinate_lookup_mismatches':mismatch,
            'subject_roi_values_checked':int(values.size),'max_subject_csv_error':maximum,
            'group_roi_rows_checked':len(rows),'pointwise_bootstrap_recomputed':1000,
            'background':args.background,'visually_reviewed':False}
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
