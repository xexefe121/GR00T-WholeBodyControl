"""Synthetic array tests only; no task state/model/engine execution."""
import unittest
import numpy as np
from diagnose_fixed_map import GROUPS,grouped,committed_target

class GroupsTest(unittest.TestCase):
    def test_axes_partition(self):
        self.assertEqual([i for _,a,b in GROUPS for i in range(a,b)],list(range(58)))
        for col in range(58):
            gain=np.ones((23,58));v=np.zeros(58);v[col]=2
            parts,other,vel,error=grouped(gain,v)
            self.assertEqual(np.count_nonzero(parts[:,0]),1)
            np.testing.assert_array_equal(other+vel,np.full(23,2.))
            self.assertEqual(error,0.)
    def test_cancellation_is_preserved(self):
        gain=np.ones((23,58));v=np.zeros(58);v[0]=1;v[35]=-1
        _,other,vel,_=grouped(gain,v)
        np.testing.assert_array_equal(other+vel,np.zeros(23))
        self.assertGreater(np.linalg.norm(other),0.)
    def test_nested_clips_original_order(self):
        gain=np.zeros((23,58));gain[:,0]=2
        target,raw,correction,preclip=committed_target(lambda x,y:np.ones(58),np.zeros(59),np.full(23,.05),gain,np.zeros(30),np.zeros(29),np.tile([-.12,.12],(23,1)))
        np.testing.assert_array_equal(raw,np.full(23,2.))
        np.testing.assert_array_equal(correction,np.full(23,.1))
        np.testing.assert_array_equal(target,np.full(23,.12))
    def test_same_chart_delta_differs_from_different_chart(self):
        # Linear group changes must subtract same-chart tangent coordinates, not unrelated chart values.
        gain=np.arange(23*58,dtype=np.float64).reshape(23,58)/100
        actual=np.linspace(-.1,.2,58);nominal=np.linspace(.02,.05,58)
        parts,_,_,_=grouped(gain,actual-nominal)
        np.testing.assert_allclose(parts.sum(axis=0),gain@actual-gain@nominal,rtol=1e-13,atol=1e-13)

if __name__=='__main__':unittest.main()
