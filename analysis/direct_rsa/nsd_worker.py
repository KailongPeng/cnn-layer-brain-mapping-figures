"""NSD direct RSA with the same full-Cube BrainIAK/MPI scoring as synthesized data."""
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
from scipy.stats import pearsonr

from .worker import score_searchlight


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--job-index',type=int,required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--check-centers',type=int,default=0)
    args=p.parse_args();comm=MPI.COMM_WORLD;rank=comm.rank
    data=mask=models=affine=reference=None;error=None;inputs=[]
    manifest_path=Path(args.manifest);manifest=json.loads(manifest_path.read_text())
    job=manifest['jobs'][args.job_index];output=Path(args.output_dir)
    if rank==0:
        try:
            if not manifest['passed'] or manifest['radius']!=3:raise ValueError('Unverified NSD manifest')
            if output.exists():raise FileExistsError(str(output))
            if hashlib.sha256(Path(job['model_rsms']).read_bytes()).hexdigest()!=job['model_sha256']:
                raise ValueError('NSD model changed')
            models=np.load(job['model_rsms'],allow_pickle=False)
            image=nib.load(job['mask']);mask=image.get_fdata()>0;affine=image.affine
            arrays=[]
            for path in [job['mask']]+job['brain_images']:
                stat=Path(path).stat();inputs.append({'path':path,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
            for path in job['brain_images']:
                image=nib.load(path)
                if image.shape!=mask.shape+(3,) or not np.allclose(image.affine,affine,atol=1e-4):
                    raise ValueError('NSD shape/affine changed')
                arrays.append(image.get_fdata().mean(axis=3))
            data=np.stack(arrays,axis=-1)
            for axis in range(3):
                edge=[slice(None)]*3;edge[axis]=slice(0,3);mask[tuple(edge)]=False
                edge[axis]=slice(-3,None);mask[tuple(edge)]=False
            if args.check_centers:
                candidates=np.argwhere(mask)
                selected=np.random.default_rng(20261108).choice(len(candidates),args.check_centers,replace=False)
                points=candidates[selected];mask=np.zeros(mask.shape,dtype=bool)
                reference=[];tri=np.triu_indices(16,1)
                for coord in points:
                    xyz=tuple(coord);mask[xyz]=True
                    slices=tuple(slice(int(k)-3,int(k)+4) for k in coord)
                    patterns=data[slices].reshape(-1,16).T
                    brain=np.corrcoef(patterns)[tri]
                    reference.append([pearsonr(model[tri],brain)[0] for model in models])
        except Exception as e:error=repr(e)
    error=comm.bcast(error,root=0)
    if error:raise RuntimeError(error)
    mask=comm.bcast(mask,root=0)
    light=Searchlight(sl_rad=3,max_blk_edge=5,shape=Cube)
    light.distribute([data],mask);light.broadcast((models,None) if rank==0 else None)
    start=time.monotonic();result=light.run_searchlight(score_searchlight,pool_size=1)
    if rank!=0:return
    report={'subject':job['subject'],'which_set':job['which_set'],'image_ids':job['image_ids'],
            'model_sha256':job['model_sha256'],'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'inputs':inputs,'seconds':time.monotonic()-start,'mpi_ranks':comm.size,
            'requested_centers':int(mask.sum()),'radius':3,'shape':'Cube',
            'brain_input_policy':manifest['brain_input_policy'],'validation_only':bool(args.check_centers),
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()}
    if args.check_centers:
        actual=np.asarray([result[tuple(coord)] for coord in points])
        expected=np.asarray(reference)
        finite=np.isfinite(actual)&np.isfinite(expected)
        if finite.sum()<12*16:raise ValueError('Insufficient finite pilot comparisons')
        report.update({'max_absolute_error':float(np.max(np.abs(actual[finite]-expected[finite]))),
                       'finite_layer_point_comparisons':int(finite.sum()),
                       'passed':bool(np.array_equal(np.isfinite(actual),np.isfinite(expected)) and
                                     np.allclose(actual,expected,atol=1e-10,rtol=1e-10,equal_nan=True))})
        if not report['passed']:raise ValueError('NSD MPI differs from independent SciPy calculation')
    else:
        maps=np.full(mask.shape+(12,),np.nan,dtype=np.float32)
        for coord in np.argwhere(mask):
            xyz=tuple(coord)
            if result[xyz] is not None:maps[xyz]=result[xyz]
        if np.isinf(maps).any() or np.any(np.abs(maps[np.isfinite(maps)])>1):raise ValueError('Invalid NSD r range')
        valid=np.isfinite(maps).all(-1)
        report['valid_centers']=int(valid.sum())
        output.mkdir(parents=True,exist_ok=False)
        for layer in range(12):nib.save(nib.Nifti1Image(maps[...,layer],affine),str(output/('layer%02d_r.nii.gz'%(layer+1))))
        nib.save(nib.Nifti1Image(valid.astype(np.uint8),affine),str(output/'valid_coverage.nii.gz'))
    output.mkdir(parents=True,exist_ok=not args.check_centers)
    (output/'qc.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
