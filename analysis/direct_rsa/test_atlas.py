import unittest
import numpy as np
from analysis.direct_rsa.atlas import fieldtrip_to_nibabel_affine


class AtlasTests(unittest.TestCase):
    def test_same_world_point_under_matlab_and_numpy_indexing(self):
        transform=np.asarray([[-1,0,0,91],[0,2,0,-128],[0,0,3,-75],[0,0,0,1]],dtype=float)
        converted=fieldtrip_to_nibabel_affine(transform)
        for index in [np.array([0,0,0]),np.array([10,31,5]),np.array([180,100,19])]:
            np.testing.assert_allclose(converted@np.r_[index,1],transform@np.r_[index+1,1])
        self.assertFalse(np.allclose(converted,transform))

    def test_reject_invalid_homogeneous_matrix(self):
        transform=np.eye(4);transform[3,0]=1
        with self.assertRaises(ValueError):fieldtrip_to_nibabel_affine(transform)


if __name__=='__main__':unittest.main()
