"""Convert FieldTrip/MATLAB one-based voxel transforms for NiBabel arrays."""
import numpy as np


def fieldtrip_to_nibabel_affine(transform):
    transform=np.asarray(transform,dtype=np.float64)
    if transform.shape!=(4,4) or not np.isfinite(transform).all():
        raise ValueError('Expected finite 4x4 FieldTrip transform')
    if not np.allclose(transform[3],[0,0,0,1]):raise ValueError('Invalid homogeneous transform')
    shift=np.eye(4);shift[:3,3]=1
    return transform@shift
