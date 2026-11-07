import itertools
import unittest
import numpy as np
from scipy.stats import ttest_1samp
from analysis.direct_rsa.inference import t_from_sums,max_t_pvalues


class InferenceTests(unittest.TestCase):
    def test_every_sign_pattern_matches_independent_student_t(self):
        values=np.random.default_rng(24).normal(size=(5,7))
        squares=(values*values).sum(axis=0)
        signs=np.asarray(list(itertools.product([-1,1],repeat=5)))
        actual=t_from_sums(signs@values,squares[None,:],5)
        expected=np.asarray([ttest_1samp(values*row[:,None],0,axis=0).statistic for row in signs])
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        null=np.max(np.abs(expected),axis=1)
        observed=ttest_1samp(values,0,axis=0).statistic
        reference=np.asarray([(1+np.sum(null>=abs(x)))/(1+len(null)) for x in observed])
        np.testing.assert_allclose(max_t_pvalues(observed,null),reference)

    def test_corrected_p_never_zero_and_counts_ties(self):
        np.testing.assert_allclose(max_t_pvalues(np.asarray([2,10]),np.asarray([1,2,3])),[.75,.25])


if __name__=='__main__':unittest.main()
