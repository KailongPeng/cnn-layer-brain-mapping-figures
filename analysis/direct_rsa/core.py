"""Signed direct Pearson RSA, adapted from historical mapping.py:ModelSim.

Source: KailongPeng/localize_paper, c94b5dc (2024-05-03),
preprocess/glm/prototype/link/scripts/mapping.py. No fitted regression,
cross-subject prediction, squaring, or permutation normalization is applied.
"""
import numpy as np


def validate_indices(indices, n_images):
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or len(indices) < 3:
        raise ValueError('Select at least three images')
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n_images):
        raise ValueError('Image indices must be unique and within bounds')
    return indices


def direct_pearson(model_rsms, patterns, indices=None):
    """Return one signed r per model layer.

    model_rsms: layers x images x images; patterns: images x voxels.
    Both sides use the same optional zero-based image indices. Undefined
    correlations remain NaN, rather than being silently replaced by zero.
    """
    models = np.asarray(model_rsms, dtype=np.float64)
    bold = np.asarray(patterns, dtype=np.float64)
    if models.ndim != 3 or models.shape[1] != models.shape[2] or bold.ndim != 2:
        raise ValueError('Expected model layers x images x images and brain images x voxels')
    if bold.shape[0] != models.shape[1] or bold.shape[1] < 2:
        raise ValueError('Brain and model image counts must match; at least two voxels required')
    ix = validate_indices(np.arange(bold.shape[0]) if indices is None else indices, bold.shape[0])
    models = models[:, ix[:, None], ix[None, :]]
    bold = bold[ix]
    tri = np.triu_indices(len(ix), 1)
    if not np.isfinite(models).all() or not np.isfinite(bold).all():
        return np.full(models.shape[0], np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        brain = np.corrcoef(bold)[tri]
    if not np.isfinite(brain).all():
        return np.full(models.shape[0], np.nan)
    vectors = models[:, tri[0], tri[1]]
    x = vectors - vectors.mean(axis=1, keepdims=True)
    y = brain - brain.mean()
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y)
    out = np.full(models.shape[0], np.nan)
    np.divide(x @ y, denom, out=out, where=denom > 0)
    return np.clip(out, -1.0, 1.0)


def paired_image_halves(n_images=16, n_splits=100, seed=20261107):
    """Reproducible complementary splits; not independent experiments."""
    if n_images < 6 or n_images % 2:
        raise ValueError('Require an even image count of at least six')
    rng = np.random.default_rng(seed)
    return [(np.sort(p[:n_images//2]), np.sort(p[n_images//2:]))
            for p in (rng.permutation(n_images) for _ in range(n_splits))]
