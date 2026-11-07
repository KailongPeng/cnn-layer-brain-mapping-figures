"""Check scientific aggregation behavior for missing and signed correlations."""
import unittest
import numpy as np
from analysis.direct_rsa.group import fisher_mean


class GroupAggregationTests(unittest.TestCase):
    def test_missing_subjects_do_not_become_zero_effects(self):
        mean, n = fisher_mean([[0.8, np.nan, -0.8, np.nan],
                               [np.nan, 0.2, 0.8, np.nan]])
        np.testing.assert_allclose(mean[:3], [0.8, 0.2, 0], atol=1e-14)
        np.testing.assert_array_equal(n, [1, 1, 2, 0])
        self.assertTrue(np.isnan(mean[3]))

    def test_uses_fisher_mean_and_preserves_sign(self):
        mean, n = fisher_mean([[0.1, -0.1], [0.9, -0.9]])
        expected = np.tanh((np.arctanh(0.1)+np.arctanh(0.9))/2)
        np.testing.assert_allclose(mean, [expected, -expected])
        self.assertNotAlmostEqual(mean[0], 0.5)

    def test_range_and_endpoints(self):
        mean, n = fisher_mean([[1, -1], [0.5, -0.5]])
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue((np.abs(mean)<1).all())
        with self.assertRaises(ValueError):
            fisher_mean([[1.1]])


if __name__ == '__main__':
    unittest.main()
