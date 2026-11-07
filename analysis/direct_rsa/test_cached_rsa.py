import unittest
import numpy as np
from analysis.direct_rsa.cached_rsa import score_cached,subset_pair_indices
from analysis.direct_rsa.core import direct_pearson,paired_image_halves


class CachedRsaTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123)
        self.models=np.asarray([np.corrcoef(rng.normal(size=(16,37))) for _ in range(12)])
        self.patterns=rng.normal(size=(4,16,343))
        tri=np.triu_indices(16,1)
        self.cache=np.asarray([np.corrcoef(pattern)[tri] for pattern in self.patterns])

    def test_full_and_complementary_subsets_match_raw_patterns(self):
        indices=[np.arange(16)]
        for a,b in paired_image_halves(n_splits=8):indices.extend([a,b])
        indices.append(np.array([15,2,7,0,11,3,8,1]))
        for ix in indices:
            expected=np.asarray([direct_pearson(self.models,pattern,ix) for pattern in self.patterns])
            np.testing.assert_allclose(score_cached(self.models,self.cache,ix),expected,atol=1e-13)

    def test_invalid_unselected_image_does_not_invalidate_other_half(self):
        patterns=self.patterns[0].copy();patterns[15]=1
        with np.errstate(invalid='ignore',divide='ignore'):
            cache=np.corrcoef(patterns)[np.triu_indices(16,1)][None,:]
        self.assertTrue(np.isnan(score_cached(self.models,cache)).all())
        ix=np.arange(8)
        np.testing.assert_allclose(score_cached(self.models,cache,ix)[0],direct_pearson(self.models,patterns,ix),atol=1e-13)

    def test_pair_lookup_and_invalid_cache(self):
        self.assertEqual(len(subset_pair_indices(np.arange(8))),28)
        with self.assertRaises(ValueError):score_cached(self.models,np.ones((1,119)))
        result=score_cached(self.models,np.ones((2,120)))
        self.assertTrue(np.isnan(result).all())


if __name__=='__main__':unittest.main()
