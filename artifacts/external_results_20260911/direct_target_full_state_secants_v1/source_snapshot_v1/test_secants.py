"""Synthetic source/math tests; no actual probes, models or native steps."""
import ast
from pathlib import Path
import unittest
import numpy as np
from direct_features import DirectFeatures
from secant_math import GROUPS,KEPT,axis_metadata,core_functions,perturb,committed_target,validate_state,validate_tangent,cell_ids

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py'

class SecantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.diff,cls.mul,cls.exp=staticmethod(core_functions(CORE)[0]),staticmethod(core_functions(CORE)[1]),staticmethod(core_functions(CORE)[2])
    def state(self):
        q=np.zeros(30);q[2]=.75;q[3]=1
        return q,np.zeros(29),np.r_[q,np.zeros(29)]
    def test_axis_metadata(self):
        radii,group,units=axis_metadata(np.arange(20,43))
        self.assertEqual(radii.shape,(58,));self.assertEqual(len(units),58)
        np.testing.assert_array_equal(radii[35:],.01*np.arange(20,43))
        np.testing.assert_array_equal(np.bincount(group),[3,3,23,3,3,23])
        self.assertEqual(KEPT.shape,(1000,))
    def test_old_velocity_expression_bitexact(self):
        q,v,p=self.state();v[:]=np.linspace(-3,4,29);p=np.r_[q,v]
        caps=np.linspace(20,37,23);r,_,_=axis_metadata(caps)
        for j in range(23):
            for sign in (-1.,1.):
                expected=v.copy();expected[6+j]+=sign*.01*caps[j]
                a,b=perturb(q,v,p,35+j,sign,r[35+j],self.diff,self.mul,self.exp)
                self.assertEqual(a.tobytes(),q.tobytes());self.assertEqual(b.tobytes(),expected.tobytes())
    def test_all_axes_in_original_chart(self):
        q,v,p=self.state();p[3:7]=self.exp(np.array([.2,-.1,.4]));q[3:7]=self.mul(p[3:7],self.exp(np.array([.04,-.02,.01])))
        r,_,_=axis_metadata(np.full(23,30.))
        for axis in range(58):
            for sign in (-1.,1.):
                a,b=perturb(q,v,p,axis,sign,r[axis],self.diff,self.mul,self.exp)
                validate_tangent(self.diff,p,q,v,a,b,axis,sign*r[axis])
                self.assertAlmostEqual(np.linalg.norm(a[3:7]),1.,places=14)
    def test_nominal_state_not_mutated(self):
        q,v,p=self.state();qb=q.tobytes();vb=v.tobytes();pb=p.tobytes()
        perturb(q,v,p,3,1.,.01,self.diff,self.mul,self.exp)
        self.assertEqual((q.tobytes(),v.tobytes(),p.tobytes()),(qb,vb,pb))
    def test_chart_boundary_is_rejected(self):
        q,v,p=self.state();q[3:7]=self.exp(np.array([np.pi-.001,0.,0.]))
        with self.assertRaisesRegex(ValueError,'chart'):perturb(q,v,p,3,1.,.01,self.diff,self.mul,self.exp)
    def test_no_position_clamp(self):
        q,v,p=self.state();q[7]=.099
        a,b=perturb(q,v,p,6,1.,.01,self.diff,self.mul,self.exp)
        self.assertGreater(a[7],.1)
        with self.assertRaisesRegex(ValueError,'position'):validate_state(a,b,np.tile([-.1,.1],(23,1)),np.full(23,30.))
    def test_bad_radius_rejected(self):
        q,v,p=self.state()
        for r in (0.,-1.,np.nan):
            with self.assertRaises(ValueError):perturb(q,v,p,0,1.,r,self.diff,self.mul,self.exp)
    def test_nested_clipped_map(self):
        q,v,p=self.state();q[7]=.02;gain=np.zeros((23,58));gain[:,6]=100
        t,raw,c,pre=committed_target(self.diff,p,np.full(23,.02),gain,q,v,np.tile([-.11,.11],(23,1)))
        np.testing.assert_array_equal(raw,np.full(23,2.));np.testing.assert_array_equal(c,np.full(23,.1));np.testing.assert_array_equal(t,np.full(23,.11))
    def test_nominal_and_velocity_centers_all_cells(self):
        d=np.repeat(np.arange(3),1019);c=np.tile(np.arange(250,1269),3)
        phase,cell=cell_ids(d,c)
        np.testing.assert_array_equal(np.bincount(cell),np.tile([100,819,100],3))
        self.assertEqual(int((np.bincount(cell)*58*2).sum()),354612)
    def test_committed_map_exact_source_ast(self):
        old=ast.parse((ROOT/'velocity_chord_student_v1/source_snapshot_v2/saved_committed_map.py').read_text())
        new=ast.parse(Path(__file__).with_name('secant_math.py').read_text())
        get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='committed_target')
        self.assertEqual(ast.dump(get(old),include_attributes=False),ast.dump(get(new),include_attributes=False))
    def test_goal_heading_recomputed(self):
        n=40;quats=np.zeros((n,3,4));quats[:,:,0]=1
        positions=np.zeros((n,3,3));positions[:,:,0]=1.;positions[:,:,2]=.75
        motion={'joint_pos':np.zeros((n,23)),'joint_vel':np.zeros((n,23)),
            'body_pos_w':positions,'body_quat_w':quats,'body_lin_vel_w':np.ones((n,3,3))*.1,'body_ang_vel_w':np.zeros((n,3,3))}
        original={'source_task_quaternion_wxyz':quats,'source_task_position_w':positions}
        contract={'default_q':np.zeros(23),'body_names':['root','left_ankle_roll_link','right_ankle_roll_link']}
        builder=DirectFeatures(motion,original,contract);q,v,p=self.state()
        center=builder(q,v,1)
        q2,v2=perturb(q,v,p,5,1.,.01,self.diff,self.mul,self.exp);changed=builder(q2,v2,1)
        self.assertFalse(np.array_equal(center[56:],changed[56:]))
        q3,v3=perturb(q,v,p,0,1.,.001,self.diff,self.mul,self.exp);translated=builder(q3,v3,1)
        self.assertFalse(np.array_equal(center[56:],translated[56:]))
        self.assertEqual(center.dtype,np.float32);self.assertEqual(center.shape,(1000,))

if __name__=='__main__':unittest.main()
