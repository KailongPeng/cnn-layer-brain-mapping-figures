import unittest
import numpy as np
from scipy.stats import pearsonr
from analysis.direct_rsa.noise_ceiling_cached import ceiling_correlations


class NoiseCeilingTests(unittest.TestCase):
    def test_matches_explicit_left_out_and_inclusive_means(self):
        x=np.random.default_rng(31).normal(size=(5,3,120))
        low,high=ceiling_correlations(x)
        for subject in range(5):
            other=np.delete(x,subject,axis=0).mean(axis=0)
            for center in range(3):
                self.assertAlmostEqual(low[subject,center],pearsonr(x[subject,center],other[center])[0],places=13)
                self.assertAlmostEqual(high[subject,center],pearsonr(x[subject,center],x.mean(axis=0)[center])[0],places=13)


if __name__=='__main__':unittest.main()
