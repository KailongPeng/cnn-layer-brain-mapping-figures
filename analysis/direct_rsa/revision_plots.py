"""Render derived review figures on Milgram; no data is transferred by this module."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def save(fig,path):
    fig.savefig(str(path.with_suffix('.png')),dpi=140)
    fig.savefig(str(path.with_suffix('.pdf')))
    plt.close(fig)


def split_plots(root):
    source=root/'image_splits/summary';output=root/'image_splits/plots'
    if output.exists():raise FileExistsError(str(output))
    if not json.loads((root/'image_splits/independent_qc.json').read_text())['passed']:raise ValueError('Split QC not passed')
    matrices=np.load(str(source/'all_cross_half_spatial_r.npy'),allow_pickle=False)
    output.mkdir(parents=True,exist_ok=False)
    fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
    im=ax.imshow(matrices.mean(0),vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(12));ax.set_yticks(range(12));ax.set_xticklabels(range(1,13));ax.set_yticklabels(range(1,13))
    ax.set_xlabel('Half B: model layer');ax.set_ylabel('Half A: model layer')
    ax.set_title('Mean cross-half spatial correlation\n100 draws; 98 distinct unordered 8/8 partitions')
    fig.colorbar(im,ax=ax,label='Spatial Pearson r');save(fig,output/'mean_cross_half_matrix')
    diagonal=np.diagonal(matrices,axis1=1,axis2=2)
    other=(matrices.sum(1)+matrices.sum(2)-2*diagonal)/22
    difference=diagonal-other
    fig,axes=plt.subplots(2,1,figsize=(10,8),constrained_layout=True)
    x=np.arange(1,13)
    for values,label,color in [(diagonal,'Same layer','#126ba0'),(other,'Different layers (direction-balanced mean)','#d57818')]:
        q=np.quantile(values,[.025,.5,.975],axis=0)
        axes[0].plot(x,q[1],marker='o',label=label,color=color)
        axes[0].fill_between(x,q[0],q[2],color=color,alpha=.18)
    axes[0].set_ylabel('Spatial Pearson r');axes[0].legend(fontsize=9)
    axes[0].set_title('Lines: across-draw median; bands: 2.5-97.5% split quantiles (not confidence intervals)')
    axes[1].boxplot([difference[:,i] for i in range(12)],positions=x,showfliers=False)
    axes[1].axhline(0,color='black',linestyle='--',linewidth=.8)
    axes[1].set_ylabel('Same minus different layer r');axes[1].set_xlabel('Model layer')
    for ax in axes:ax.set_xticks(x);ax.set_xlim(.5,12.5)
    fig.suptitle('Image subset robustness within the existing 16 images | 30 participants')
    save(fig,output/'same_vs_different_layers')
    (output/'qc.json').write_text(json.dumps({'generated':True,'visually_reviewed':False,'split_quantiles_are_not_confidence_intervals':True},indent=2))


def annotated_synth(root):
    inference=root/'inference';output=root/'inference/plots'
    if output.exists():raise FileExistsError(str(output))
    if not json.loads((inference/'independent_qc.json').read_text())['passed']:raise ValueError('Inference QC not passed')
    mask_img=nib.as_closest_canonical(nib.load(str(root/'group/common_all30_mask.nii.gz')))
    mask=mask_img.get_fdata()>0;affine=mask_img.affine
    zs=[-8,8,24];indices=[int(round(nib.affines.apply_affine(np.linalg.inv(affine),[0,0,z])[2])) for z in zs]
    fig,axes=plt.subplots(12,3,figsize=(11,27),constrained_layout=True)
    for layer in range(1,13):
        data=nib.as_closest_canonical(nib.load(str(root/'group'/('layer%02d_mean_r.nii.gz'%layer)))).get_fdata()
        sig=nib.as_closest_canonical(nib.load(str(inference/('layer%02d_significant_fwe.nii.gz'%layer)))).get_fdata()>0
        for column,k in enumerate(indices):
            ax=axes[layer-1,column]
            im=ax.imshow(np.ma.masked_where(~mask[:,:,k].T,data[:,:,k].T),origin='lower',cmap='RdBu_r',vmin=-.15,vmax=.15)
            if sig[:,:,k].any() and not sig[:,:,k].all():ax.contour(sig[:,:,k].T,levels=[.5],colors='black',linewidths=.5)
            ax.set_title('Layer %02d | z=%d mm'%(layer,zs[column]),fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=axes,shrink=.45,label='Model-brain r: Fisher-aggregated across 30 subjects')
    fig.suptitle('Color: unthresholded direct r | Black contours: two-sided FWE p <= .05\n'
                 '10,000 subject sign flips; max-|t| jointly across all 12 layers and common space',fontsize=12)
    output.mkdir(parents=True,exist_ok=False);save(fig,output/'group_r_with_new_significance')
    (output/'qc.json').write_text(json.dumps({'generated':True,'visually_reviewed':False,'r_color_limit':.15,'significance_source':'new joint max-|t| subject sign-flip inference'},indent=2))


def compare_datasets(root):
    if not json.loads((root/'nsd/independent_qc.json').read_text())['passed']:raise ValueError('NSD QC not passed')
    output=root/'nsd/comparison_plots'
    if output.exists():raise FileExistsError(str(output))
    synth_mask=nib.as_closest_canonical(nib.load(str(root/'group/common_all30_mask.nii.gz')))
    nsd_mask=nib.as_closest_canonical(nib.load(str(root/'nsd/group/common_all8_all16_mask.nii.gz')))
    if synth_mask.shape!=nsd_mask.shape or not np.allclose(synth_mask.affine,nsd_mask.affine):raise ValueError('Comparison space mismatch')
    common=(synth_mask.get_fdata()>0)&(nsd_mask.get_fdata()>0)
    if common.sum()<3:raise ValueError('No dataset common coverage')
    synth=[];nsd=[]
    for layer in range(1,13):
        synth.append(nib.as_closest_canonical(nib.load(str(root/'group'/('layer%02d_mean_r.nii.gz'%layer)))).get_fdata())
        nsd.append(nib.as_closest_canonical(nib.load(str(root/'nsd/group'/('layer%02d_mean_r.nii.gz'%layer)))).get_fdata())
    vmax=np.ceil(max(float(np.abs(x[common]).max()) for x in synth+nsd)*20)/20
    indices=[int(round(nib.affines.apply_affine(np.linalg.inv(synth_mask.affine),[0,0,z])[2])) for z in [-8,8,24]]
    fig,axes=plt.subplots(12,6,figsize=(20,27),constrained_layout=True)
    for layer in range(12):
        for dataset,collection in enumerate([synth,nsd]):
            for column,k in enumerate(indices):
                ax=axes[layer,dataset*3+column]
                im=ax.imshow(np.ma.masked_where(~common[:,:,k].T,collection[layer][:,:,k].T),origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
                ax.set_title('%s L%02d | z=%d'%(['Synth','NSD'][dataset],layer+1,[-8,8,24][column]),fontsize=9)
                ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=axes,shrink=.45,label='Direct model-brain Pearson r; common scale')
    fig.suptitle('Descriptive comparison | Synth: 30 subjects, 16 images | NSD: 8 subjects, 16 image sets\n'
                 'Common valid space; participant/repetition/stimulus differences limit causal interpretation',fontsize=13)
    output.mkdir(parents=True,exist_ok=False);save(fig,output/'synth_nsd_common_scale')
    (output/'qc.json').write_text(json.dumps({'generated':True,'visually_reviewed':False,'common_voxels':int(common.sum()),'color_limit':float(vmax),'descriptive_only':True},indent=2))


def noise_ceiling_plot(root):
    source=root/'noise_ceiling';output=source/'plots'
    if output.exists():raise FileExistsError(str(output))
    if not json.loads((source/'independent_qc.json').read_text())['passed']:raise ValueError('Noise ceiling QC not passed')
    ceiling=list(csv.DictReader((source/'roi_noise_ceiling.csv').open()))
    models=list(csv.DictReader((root/'roi/group_roi_r.csv').open()))
    fig,axes=plt.subplots(4,4,figsize=(15,11),constrained_layout=True)
    for index,row in enumerate(ceiling):
        values=sorted([r for r in models if r['roi']==row['roi']],key=lambda r:int(r['layer']))
        ax=axes.flat[index]
        ax.axhspan(float(row['lower_mean_r']),float(row['upper_mean_r']),color='gray',alpha=.22)
        ax.plot(range(1,13),[float(r['mean_subject_roi_r']) for r in values],marker='o',markersize=3,color='#126ba0')
        ax.axhline(0,color='black',linewidth=.6);ax.set_title(row['roi'])
        ax.set_xticks([1,4,8,12]);ax.tick_params(axis='x',labelbottom=True)
        ax.set_xlabel('Model layer');ax.set_ylabel('Mean raw Pearson r')
    for index in range(len(ceiling),16):axes.flat[index].set_visible(False)
    fig.suptitle('Current 16-image synthesized data: model r and brain-RSM reliability\n'
                 'Identical full-Cube inputs and common coverage; gray band is not a confidence interval')
    output.mkdir(parents=True,exist_ok=False);save(fig,output/'roi_model_r_and_noise_ceiling')
    (output/'qc.json').write_text(json.dumps({'generated':True,'visually_reviewed':False,'change':'show layer tick labels on every ROI panel; no numerical recomputation'},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['splits','inference','nsd','noise_ceiling']);p.add_argument('--root',required=True)
    args=p.parse_args();root=Path(args.root)
    if args.mode=='splits':split_plots(root)
    elif args.mode=='inference':annotated_synth(root)
    elif args.mode=='nsd':compare_datasets(root)
    else:noise_ceiling_plot(root)


if __name__=='__main__':main()
