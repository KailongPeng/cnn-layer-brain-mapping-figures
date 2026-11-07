"""Score image subsets from cached full brain RSMs, without changing the estimand."""
import numpy as np
from .core import validate_indices


def subset_pair_indices(indices,n_images=16):
    ix=validate_indices(indices,n_images)
    tri=np.triu_indices(n_images,1)
    lookup=np.full((n_images,n_images),-1,dtype=int)
    lookup[tri]=np.arange(len(tri[0]));lookup[(tri[1],tri[0])]=np.arange(len(tri[0]))
    within=np.triu_indices(len(ix),1)
    return lookup[ix[within[0]],ix[within[1]]]


def score_cached(models,brain_upper,indices=None):
    """brain_upper: centers x full-RSM upper triangle; return centers x layers."""
    models=np.asarray(models,dtype=np.float64)
    brain=np.asarray(brain_upper,dtype=np.float64)
    if models.ndim!=3 or models.shape[1]!=models.shape[2]:raise ValueError('Invalid model shape')
    n=models.shape[1]
    if brain.ndim!=2 or brain.shape[1]!=n*(n-1)//2:raise ValueError('Invalid cached brain shape')
    ix=validate_indices(np.arange(n) if indices is None else indices,n)
    pair_indices=subset_pair_indices(ix,n)
    subset=models[:,ix[:,None],ix[None,:]]
    tri=np.triu_indices(len(ix),1)
    x=subset[:,tri[0],tri[1]]
    if not np.isfinite(x).all():return np.full((len(brain),len(models)),np.nan)
    x=x-x.mean(axis=1,keepdims=True)
    y=brain[:,pair_indices]
    good=np.isfinite(y).all(axis=1)
    out=np.full((len(brain),len(models)),np.nan)
    if good.any():
        y=y[good]-y[good].mean(axis=1,keepdims=True)
        denom=np.linalg.norm(y,axis=1)[:,None]*np.linalg.norm(x,axis=1)[None,:]
        values=np.full(denom.shape,np.nan)
        np.divide(y@x.T,denom,out=values,where=denom>0)
        out[good]=np.clip(values,-1,1)
    return out
