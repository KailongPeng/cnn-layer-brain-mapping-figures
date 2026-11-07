"""Validate historical NSD image identities and make an explicit 128-job manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import nibabel as nib
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.source);output=Path(args.output)
    if output.exists():raise FileExistsError(str(output))
    jobs=[];reference=None
    for which_set in range(1,17):
        model=root/'modelData/data'/('ActualFingerprints_whichSet_%d.npy'%which_set)
        metadata=model.with_suffix('.pkl')
        meta=pickle.load(metadata.open('rb'))[0]
        models=np.load(str(model),allow_pickle=False)
        if models.shape!=(12,16,16) or not np.isfinite(models).all():raise ValueError('Invalid NSD model')
        if not np.allclose(np.asarray(meta['ActualFingerprints']).reshape(models.shape),models):
            raise ValueError('Model array differs from metadata')
        manifest=root/'findLayerInBrain_leaveOneSubOutReg/searchlight_score_linearModel'/('sl_jobArray_whichSet_1_%d.npy'%(which_set-1))
        records=np.load(str(manifest),allow_pickle=True).item()
        ids=[int(i) for i in meta['setOf16Images']]
        if len(ids)!=16 or len(set(ids))!=16:raise ValueError('Invalid NSD image IDs')
        for sub in range(1,9):
            subject='sub_i%d'%sub
            selected=[x for x in records.values() if x[0]==subject]
            if len(selected)!=12 or any(int(x[4])!=which_set or list(x[5])!=ids for x in selected):
                raise ValueError('NSD manifest/image identity mismatch')
            mask=root/'brainData/data'/('%s_mask_whichSet%d_2mm.nii.gz'%(subject,which_set))
            images=[root/'brainData/data'/('%s_imgID%d_2mm.nii.gz'%(subject,image)) for image in ids]
            for path in [mask]+images:
                image=nib.load(str(path))
                if reference is None:reference=(image.shape[:3],image.affine)
                expected_shape=reference[0] if path==mask else reference[0]+(3,)
                if image.shape!=expected_shape or not np.allclose(image.affine,reference[1],atol=1e-4):
                    raise ValueError('NSD input space/repetition mismatch: '+str(path))
            jobs.append({'subject':subject,'which_set':which_set,'image_ids':ids,
                         'brain_images':[str(x) for x in images],'mask':str(mask),
                         'model_rsms':str(model),'model_sha256':sha(model),
                         'historical_manifest':str(manifest),'historical_manifest_sha256':sha(manifest),
                         'model_metadata':str(metadata),'model_metadata_sha256':sha(metadata)})
    report={'passed':True,'radius':3,'shape':'Cube','n_jobs':len(jobs),'jobs':jobs,
            'brain_input_policy':'arithmetic mean of three repetitions per image before brain RSM',
            'identity':'model metadata ordered IDs match every subject/layer historical job entry',
            'array_geometry':list(reference[0]),'affine':reference[1].tolist()}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'passed':True,'n_jobs':len(jobs),'brain_files_checked':len(jobs)*16}))


if __name__=='__main__':main()
