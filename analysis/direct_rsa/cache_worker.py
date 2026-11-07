"""Cache brain RSMs on fixed all-subject coverage, checking all original r values."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
from mpi4py import MPI
from brainiak.searchlight.searchlight import Cube,Searchlight
from .cached_rsa import score_cached


def brain_rsm(data,mask,radius,broadcast):
    block=data[0]
    if block.shape[:3]!=(7,7,7):return np.full(120,np.nan)
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.corrcoef(block.reshape(-1,16).T)[np.triu_indices(16,1)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--root',required=True);p.add_argument('--subject',required=True)
    args=p.parse_args();comm=MPI.COMM_WORLD;rank=comm.rank
    root=Path(args.root);config=Path(args.config);cfg=json.loads(config.read_text())
    out=root/'brain_rsm_cache'/args.subject
    data=mask=affine=None;error=None;inputs=[]
    mask_path=root/'group/common_all30_mask.nii.gz'
    if rank==0:
        try:
            if args.subject not in cfg['subjects']:raise ValueError('Unexpected subject')
            if out.exists():raise FileExistsError(str(out))
            img=nib.load(str(mask_path));mask=img.get_fdata()>0;affine=img.affine
            subject_mask=nib.load(cfg['mask'].format(subject=args.subject))
            if subject_mask.shape!=mask.shape or not np.allclose(subject_mask.affine,affine):raise ValueError('Mask space mismatch')
            if np.any(mask&~(subject_mask.get_fdata()>0)):raise ValueError('Common mask outside subject mask')
            arrays=[]
            for i in range(1,17):
                path=Path(cfg['cope'].format(subject=args.subject,image=i));stat=path.stat()
                inputs.append({'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
                img=nib.load(str(path))
                if img.shape!=mask.shape or not np.allclose(img.affine,affine):raise ValueError('Cope space mismatch')
                arrays.append(img.get_fdata())
            data=np.stack(arrays,axis=-1)
        except Exception as e:error=repr(e)
    error=comm.bcast(error,root=0)
    if error:raise RuntimeError(error)
    mask=comm.bcast(mask,root=0)
    sl=Searchlight(sl_rad=3,max_blk_edge=5,shape=Cube)
    sl.distribute([data],mask);sl.broadcast(None)
    start=time.monotonic();result=sl.run_searchlight(brain_rsm,pool_size=1)
    if rank!=0:return
    coords=np.argwhere(mask)
    cache=np.asarray([result[tuple(coord)] for coord in coords],dtype=np.float64)
    if cache.shape!=(len(coords),120) or not np.isfinite(cache).all():raise ValueError('Invalid full-RSM common coverage cache')
    models=np.load(cfg['model_rsms'],allow_pickle=False)
    actual=np.concatenate([score_cached(models,cache[start:start+4096]) for start in range(0,len(cache),4096)])
    expected=np.asarray([nib.load(str(root/'full'/args.subject/('layer%02d_r.nii.gz'%layer))).get_fdata()[mask] for layer in range(1,13)]).T
    error=float(np.max(np.abs(actual-expected)))
    if not np.allclose(actual,expected,atol=1e-7,rtol=1e-7):raise ValueError('Cached RSM scores differ from existing full maps')
    report={'subject':args.subject,'passed':True,'centers':len(coords),'dtype':'float64',
            'r_comparisons':int(actual.size),'max_full_r_absolute_error':error,
            'coverage':'fixed 30-subject/12-layer common mask; same for all future splits',
            'mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256(Path(cfg['model_rsms']).read_bytes()).hexdigest(),
            'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'inputs':inputs,
            'seconds':time.monotonic()-start,'radius':3,'shape':'Cube','mpi_ranks':comm.size,
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()}
    out.mkdir(parents=True,exist_ok=False)
    np.save(str(out/'brain_upper.npy'),cache,allow_pickle=False)
    np.save(str(out/'coordinates.npy'),coords,allow_pickle=False)
    (out/'qc.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
