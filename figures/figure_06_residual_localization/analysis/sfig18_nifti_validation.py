"""Shared fail-loud NIfTI checks for the residual Figure S18 pipeline."""

import os
import tempfile

import nibabel as nib
import numpy as np


EXPECTED_SHAPE = (91, 109, 91)
EXPECTED_AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def validate_nifti(
    path,
    *,
    expected_shape=EXPECTED_SHAPE,
    expected_affine=EXPECTED_AFFINE,
    require_nonzero=False,
    require_binary=False,
):
    """Load the complete file and reject truncated, misregistered, or invalid maps."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) < 1024:
        raise ValueError(f"NIfTI is unexpectedly small or truncated: {path}")
    nii = nib.load(path)
    if tuple(nii.shape) != tuple(expected_shape):
        raise ValueError(f"NIfTI shape {nii.shape} != {expected_shape}: {path}")
    if not np.allclose(nii.affine, expected_affine, atol=1e-4):
        raise ValueError(f"NIfTI affine does not match the MNI 2-mm reference: {path}")
    data = nii.get_fdata(dtype=np.float64)
    if not np.all(np.isfinite(data)):
        count = int(np.sum(~np.isfinite(data)))
        raise ValueError(f"NIfTI contains {count} non-finite values: {path}")
    if require_nonzero and not np.any(data != 0):
        raise ValueError(f"NIfTI is entirely zero: {path}")
    if require_binary and not np.all(np.isin(np.unique(data), [0, 1])):
        raise ValueError(f"NIfTI is not binary 0/1: {path}")
    return nii, data


def atomic_save_nifti(
    data,
    affine,
    destination,
    *,
    require_nonzero=False,
    require_binary=False,
):
    """Write beside the destination, validate fully, then atomically publish it."""
    output_dir = os.path.dirname(destination)
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".nii.gz", dir=output_dir
    )
    os.close(fd)
    try:
        nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float64), affine), temporary)
        validate_nifti(
            temporary,
            expected_affine=affine,
            require_nonzero=require_nonzero,
            require_binary=require_binary,
        )
        os.replace(temporary, destination)
        validate_nifti(
            destination,
            expected_affine=affine,
            require_nonzero=require_nonzero,
            require_binary=require_binary,
        )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
