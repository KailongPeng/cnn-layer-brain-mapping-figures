"""BrainIAK/MPI direct RSA; optional sparse validation against the NumPy pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
from mpi4py import MPI
from brainiak.searchlight.searchlight import Cube, Searchlight

from .core import direct_pearson, validate_indices


def score_searchlight(data, mask, radius, broadcast):
    features, indices = broadcast
    block = data[0]
    if block.shape[:3] != (2*radius+1,)*3:
        return np.full(features.shape[0], np.nan)
    return direct_pearson(features, block.reshape(-1, block.shape[-1]).T, indices)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--subject',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--reference',help='Sparse validation only; never writes partial NIfTI maps')
    p.add_argument('--indices',type=int,nargs='+')
    args=p.parse_args()
    comm=MPI.COMM_WORLD; rank=comm.rank
    config_path=Path(args.config)
    cfg=json.loads(config_path.read_text())
    indices=validate_indices(list(range(16)) if args.indices is None else args.indices,16).tolist()
    out=Path(args.output_dir)
    data=mask=features=affine=None
    reference=None; error=None
    input_records=[]; model_digest=None
    if rank==0:
        try:
            if args.subject not in cfg['subjects']:
                raise ValueError('Unexpected subject')
            if out.exists():
                raise FileExistsError('Preserve existing output directory: '+str(out))
            paths=[Path(cfg['mask'].format(subject=args.subject))]
            paths += [Path(cfg['cope'].format(subject=args.subject,image=i)) for i in range(1,17)]
            for path in paths:
                stat=path.stat()
                input_records.append({'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
            model_digest=hashlib.sha256(Path(cfg['model_rsms']).read_bytes()).hexdigest()
            if not args.reference:
                provenance_path=Path(cfg['stimulus_identity_report'])
                provenance=json.loads(provenance_path.read_text())
                if not provenance['passed'] or provenance['features_sha256']!=model_digest:
                    raise ValueError('Stimulus identity gate has not passed for this model')
            image=nib.load(cfg['mask'].format(subject=args.subject))
            mask=image.get_fdata()>0; affine=image.affine
            arrays=[]
            for i in range(1,17):
                img=nib.load(cfg['cope'].format(subject=args.subject,image=i))
                if img.shape!=mask.shape or not np.allclose(img.affine,affine,atol=1e-4):
                    raise ValueError('Spatial mismatch in cope '+str(i))
                arrays.append(img.get_fdata())
            data=np.stack(arrays,axis=-1)
            features=np.load(cfg['model_rsms'],allow_pickle=False)
            if features.shape!=(12,16,16) or not np.isfinite(features).all():
                raise ValueError('Unexpected features')
            radius=int(cfg['radius'])
            for axis in range(3):
                a=[slice(None)]*3; a[axis]=slice(0,radius); mask[tuple(a)]=False
                a[axis]=slice(-radius,None); mask[tuple(a)]=False
            if args.reference:
                if args.indices is not None:
                    raise ValueError('Reference comparison requires all 16 images')
                reference=json.loads(Path(args.reference).read_text())
                digest=hashlib.sha256(Path(cfg['model_rsms']).read_bytes()).hexdigest()
                if reference['subject']!=args.subject or digest!=reference['model_sha256']:
                    raise ValueError('Reference subject/model mismatch')
                sparse=np.zeros(mask.shape,dtype=bool)
                for point in reference['points']:
                    coord=tuple(point['ijk'])
                    if not mask[coord]: raise ValueError('Reference outside valid center mask')
                    sparse[coord]=True
                mask=sparse
        except Exception as e:
            error=repr(e)
    error=comm.bcast(error,root=0)
    if error:
        raise RuntimeError(error)
    mask=comm.bcast(mask,root=0)
    sl=Searchlight(sl_rad=int(cfg['radius']),max_blk_edge=5,shape=Cube)
    sl.distribute([data],mask)
    sl.broadcast((features,args.indices) if rank==0 else None)
    start=time.monotonic()
    result=sl.run_searchlight(score_searchlight,pool_size=1)
    if rank!=0:
        return
    metadata={'subject':args.subject,'mpi_ranks':comm.size,'seconds':time.monotonic()-start,
              'requested_centers':int(mask.sum()),'radius':cfg['radius'],'shape':'Cube',
              'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'image_order_status':cfg['image_order_status'],
              'image_indices':indices,'model_sha256':model_digest,
              'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'inputs':input_records,'voxel_policy':'all voxels in full 7x7x7 Cube; mask selects centers only',
              'boundary_policy':'remove centers within radius of array edge'}
    if reference is not None:
        actual=[]; expected=[]
        for point in reference['points']:
            value=result[tuple(point['ijk'])]
            if value is None: raise RuntimeError('Searchlight omitted reference center')
            actual.append(np.asarray(value)); expected.append(point['r_16_images'])
        actual=np.asarray(actual); expected=np.asarray(expected)
        metadata.update({'validation_only':True,'centers_compared':len(actual),
                         'max_absolute_error':float(np.max(np.abs(actual-expected))),
                         'passed':bool(np.allclose(actual,expected,atol=1e-10,rtol=1e-10))})
        out.mkdir(parents=True,exist_ok=False)
        (out/'qc.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
        if not metadata['passed']: raise RuntimeError('MPI result disagrees with pilot')
    else:
        maps=np.full(mask.shape+(12,),np.nan,dtype=np.float32)
        for coord in np.argwhere(mask):
            xyz=tuple(coord)
            if result[xyz] is not None: maps[xyz]=result[xyz]
        if np.any(np.abs(maps[np.isfinite(maps)])>1+1e-6):
            raise RuntimeError('Invalid Pearson range')
        out.mkdir(parents=True,exist_ok=False)
        for layer in range(12):
            nib.save(nib.Nifti1Image(maps[...,layer],affine),str(out/('layer%02d_r.nii.gz'%(layer+1))))
        valid=np.isfinite(maps).all(-1)
        nib.save(nib.Nifti1Image(valid.astype(np.uint8),affine),str(out/'valid_coverage.nii.gz'))
        metadata.update({'validation_only':False,'valid_centers':int(valid.sum())})
        (out/'qc.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    main()
