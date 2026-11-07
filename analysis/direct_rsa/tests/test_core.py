import unittest
import numpy as np
from scipy.stats import pearsonr
from analysis.direct_rsa.core import direct_pearson, paired_image_halves


class DirectRSATest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.bold = rng.normal(size=(16, 343))
        self.models = np.stack([np.corrcoef(rng.normal(size=(16, 20))) for _ in range(12)])

    def test_matches_historical_ModelSim_and_scipy(self):
        tri = np.triu_indices(16, 1)
        brain = np.corrcoef(self.bold)[tri]
        scipy = [pearsonr(f[tri], brain)[0] for f in self.models]
        legacy = [np.corrcoef(np.vstack((f[tri], brain)))[0,1] for f in self.models]
        actual = direct_pearson(self.models, self.bold)
        np.testing.assert_allclose(actual, scipy, atol=1e-12)
        np.testing.assert_allclose(actual, legacy, atol=1e-12)

    def test_eight_images_select_both_sides(self):
        idx = np.array([0,2,4,6,8,10,12,14])
        actual = direct_pearson(self.models, self.bold, idx)
        expected = direct_pearson(self.models[:,idx[:,None],idx[None,:]], self.bold[idx])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_identical_and_reversed_geometry_keep_sign(self):
        brain = np.corrcoef(self.bold)
        np.testing.assert_allclose(direct_pearson(np.stack([brain,-brain]), self.bold), [1,-1])

    def test_constant_pattern_stays_undefined(self):
        self.bold[0] = 0
        self.assertTrue(np.isnan(direct_pearson(self.models, self.bold)).all())

    def test_halves_are_complementary_and_reproducible(self):
        a = paired_image_halves(n_splits=5)
        b = paired_image_halves(n_splits=5)
        for (x,y),(xx,yy) in zip(a,b):
            self.assertFalse(set(x)&set(y))
            self.assertEqual(set(x)|set(y),set(range(16)))
            np.testing.assert_array_equal(x,xx)
            np.testing.assert_array_equal(y,yy)


if __name__ == '__main__':
    unittest.main()
