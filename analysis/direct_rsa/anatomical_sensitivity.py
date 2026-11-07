"""Descriptive spatial comparisons within the existing, independent anatomical ROI union."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from .image_splits import cross_half_spatial


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root);output=root/'anatomical_sensitivity'
    if output.exists():raise FileExistsError(str(output))
    if not json.loads((root/'roi_corrected/independent_qc.json').read_text())['passed']:raise ValueError('Corrected ROI QC not passed')
    common_img=nib.load(str(root/'group/common_all30_mask.nii.gz'));common=common_img.get_fdata()>0
    atlas_img=nib.load(str(root/'roi_corrected/wang_atlas_mni2mm.nii.gz'))
    if atlas_img.shape!=common_img.shape or not np.allclose(atlas_img.affine,common_img.affine):raise ValueError('Sensitivity mask space mismatch')
    definitions=json.loads((root/'roi_corrected/qc.json').read_text())['definitions']
    ids=sorted({i for d in definitions for i in d['label_ids']})
    atlas=atlas_img.get_fdata();mask=common&np.isin(atlas,ids);selected=np.flatnonzero(mask[common])
    if len(selected)<100:raise ValueError('Insufficient anatomical coverage')
    matrices=[]
    for folder in sorted((root/'image_splits').glob('batch*/split*')):
        maps=np.load(str(folder/'group_r.npy'),allow_pickle=False,mmap_mode='r')
        matrices.append(cross_half_spatial(maps[0,selected,:],maps[1,selected,:]))
    matrices=np.asarray(matrices)
    if matrices.shape!=(100,12,12) or not np.isfinite(matrices).all():raise ValueError('Invalid anatomical split matrices')
    maps=np.asarray([nib.load(str(root/'group'/('layer%02d_mean_r.nii.gz'%layer))).get_fdata()[mask] for layer in range(1,13)])
    spatial=np.corrcoef(maps)
    output.mkdir(parents=True,exist_ok=False)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8),common_img.affine),str(output/'anatomical_common_mask.nii.gz'))
    np.savetxt(str(output/'full16_spatial_r.csv'),spatial,delimiter=',')
    np.save(str(output/'all_cross_half_spatial_r.npy'),matrices,allow_pickle=False)
    primary=np.load(str(root/'image_splits/summary/all_cross_half_spatial_r.npy'),allow_pickle=False)
    fig,axes=plt.subplots(2,1,figsize=(10,8),constrained_layout=True);x=np.arange(1,13)
    summaries=[]
    for source,label,color in [(primary,'Primary: whole common coverage','#126ba0'),(matrices,'Sensitivity: anatomical visual ROI union','#c27318')]:
        same=np.diagonal(source,axis1=1,axis2=2)
        other=(source.sum(1)+source.sum(2)-2*same)/22
        for values,ax,measure in [(same,axes[0],'same_layer_r'),(same-other,axes[1],'same_minus_different')]:
            q=np.quantile(values,[.025,.5,.975],axis=0)
            ax.plot(x,q[1],marker='o',color=color,label=label);ax.fill_between(x,q[0],q[2],color=color,alpha=.17)
            ax.axhline(0,color='black',linestyle='--',linewidth=.7);ax.set_xticks(x)
            for layer in range(12):summaries.append([label,measure,layer+1,float(q[1,layer]),float(q[0,layer]),float(q[2,layer])])
    axes[0].set_ylabel('Same-layer cross-half spatial r');axes[1].set_ylabel('Same minus different layer r');axes[1].set_xlabel('Model layer')
    axes[0].legend(fontsize=9);fig.suptitle('Image-split sensitivity to comparison space\nMedian and 2.5-97.5% split quantiles; descriptive, not confidence intervals')
    fig.savefig(str(output/'comparison_space_sensitivity.png'),dpi=150);fig.savefig(str(output/'comparison_space_sensitivity.pdf'));plt.close(fig)
    with (output/'layer_distribution.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['space','measure','layer','median','q025','q975']);writer.writerows(summaries)
    report={'passed':True,'descriptive_only':True,'anatomical_voxels':int(mask.sum()),'n_splits':100,
            'roi_names':[d['roi'] for d in definitions],'selection':'union of all 13 previously specified bilateral Wang ROIs, intersect fixed 30-subject common coverage',
            'layer_significance_selection':False,'primary_analysis_unchanged':True,'visually_reviewed':False}
    (output/'qc.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
