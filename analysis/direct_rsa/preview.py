"""Unthresholded, uniformly scaled descriptive previews of direct RSA maps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    source=Path(args.group_dir); output=Path(args.output_dir)
    if output.exists(): raise FileExistsError(str(output))
    qc=json.loads((source/'qc.json').read_text())
    if not qc['descriptive_only'] or qc['n_subjects']!=30:
        raise ValueError('Unexpected group estimate')
    images=[nib.as_closest_canonical(nib.load(str(source/('layer%02d_mean_r.nii.gz'%i)))) for i in range(1,13)]
    mask=nib.as_closest_canonical(nib.load(str(source/'common_all30_mask.nii.gz'))).get_fdata()>0
    maps=[image.get_fdata() for image in images]
    vmax=np.ceil(max(float(np.max(np.abs(data[mask]))) for data in maps)*20)/20
    if vmax<=0: raise ValueError('Zero color scale')
    affine=images[0].affine
    if not np.allclose(affine[:3,:3],np.diag(np.diag(affine[:3,:3]))):
        raise ValueError('Oblique images require explicit resampling')
    slices=[-8,8,24]
    z_indices=[int(round(nib.affines.apply_affine(np.linalg.inv(affine),[0,0,z])[2])) for z in slices]
    extent=[affine[0,3]-affine[0,0]/2,affine[0,3]+affine[0,0]*(mask.shape[0]-.5),
            affine[1,3]-affine[1,1]/2,affine[1,3]+affine[1,1]*(mask.shape[1]-.5)]
    fig,axes=plt.subplots(12,3,figsize=(11,27),constrained_layout=True)
    for layer,data in enumerate(maps):
        for column,k in enumerate(z_indices):
            ax=axes[layer,column]
            ax.imshow(mask[:,:,k].T,origin='lower',extent=extent,cmap='Greys',vmin=0,vmax=3)
            image=ax.imshow(np.ma.masked_where(~mask[:,:,k].T,data[:,:,k].T),origin='lower',
                            extent=extent,cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
            ax.set_title('Layer %02d | z=%d mm'%(layer+1,slices[column]),fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
            if layer==0:ax.set_xlabel('Left                                      Right',fontsize=8);ax.xaxis.set_label_position('top')
    fig.colorbar(image,ax=axes,shrink=.45,label='Model-brain Pearson r: tanh(mean(subject Fisher z))')
    fig.suptitle('Synthesized images | 30 subjects | unthresholded descriptive preview\n'
                 'Common coverage across 30 subjects and 12 layers; no significance mask',fontsize=13)
    output.mkdir(parents=True,exist_ok=False)
    fig.savefig(str(output/'group_r_axial_preview.png'),dpi=130)
    fig.savefig(str(output/'group_r_axial_preview.pdf'))
    plt.close(fig)
    spatial=np.loadtxt(str(source/'spatial_correlation_common_all30.csv'),delimiter=',')
    fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
    im=ax.imshow(spatial,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(12));ax.set_yticks(range(12))
    ax.set_xticklabels(range(1,13));ax.set_yticklabels(range(1,13))
    ax.set_xlabel('CNN layer');ax.set_ylabel('CNN layer')
    ax.set_title('Spatial Pearson between group direct-r maps\nCommon all-30 coverage | descriptive')
    fig.colorbar(im,ax=ax,label='Spatial Pearson r')
    fig.savefig(str(output/'spatial_correlation_preview.png'),dpi=180)
    fig.savefig(str(output/'spatial_correlation_preview.pdf'))
    plt.close(fig)
    (output/'qc.json').write_text(json.dumps({'descriptive_only':True,'color_limit':float(vmax),
        'axial_z_mm':slices,'display_orientation':'neurological; left on left',
        'coverage':'all 30 subjects, all 12 layers','significance_overlay':False},indent=2))


if __name__=='__main__':
    main()
