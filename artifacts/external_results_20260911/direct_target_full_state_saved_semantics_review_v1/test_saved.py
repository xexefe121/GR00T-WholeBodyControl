"""Synthetic arrays and source inspection only; no task-head/native execution."""
import unittest
from audit_saved import segment,check_inference
from saved_common import *
from fixed_maps import grouped,committed_target

def terms(q,dq,quat,omega,previous,default):
    h=history_zero();values={k:np.zeros(v.shape[1],np.float32) for k,v in h.items()};values['actions']=previous.copy()
    return np.zeros(52,np.float32),values

def synthetic(control,substeps=10,previous=None,history=None):
    previous=np.zeros(23,np.float32) if previous is None else previous.copy()
    history=history_zero() if history is None else {k:v.copy() for k,v in history.items()}
    c=dict(default_q=[0.]*23,kp=[3.]*23,training_effort=[4.]*23,joint_limits=[[-1.,1.]]*23)
    q=np.zeros(30,np.float64);q[3]=1.;dq=np.zeros(29,np.float64);full=np.zeros(291,np.float64)
    full[0]=1.;full[1:31]=q;clock=times(1.,substeps);end=full.copy();end[0]=clock[-1]
    head=np.full(23,.70000005,np.float32) if phase(control)==1 else np.zeros(23,np.float32)
    span=np.full(23,2.,np.float64);default=np.asarray(c['default_q']);kp=np.asarray(c['kp']);effort=np.asarray(c['training_effort'])
    action=np.full(23,6.,np.float32);base=default if phase(control)==1 else default+action*.25*effort/kp
    delta=span*head.astype(np.float64) if phase(control)==1 else np.zeros(23,np.float32)
    raw=base+delta;target=np.clip(raw,-1.,1.);actual=actual_action(target,default,kp,effort)
    if phase(control)==1:action=actual.copy()
    sensed,values=terms(q[7:],dq[6:],q[3:7],dq[3:6],previous,default)
    after={k:v.copy() for k,v in history.items()};advance(after,values)
    a=dict(target=target[None],qpos=np.stack([q,q]),qvel=np.stack([dq,dq]),global_control=np.array([control],np.int64),
        source_frame=np.array([min(control+11,1579)],np.int64),controller_mode=np.array([phase(control)],np.int64),physics_substeps=np.array([substeps],np.int64),
        physics_torque=np.zeros((substeps,23)),physics_qpos=np.tile(q,(substeps+1,1)),physics_qvel=np.tile(dq,(substeps+1,1)),physics_time=clock,physics_expected_time=clock.copy(),
        physics_warning_counts=np.zeros((substeps+1,8),np.int32),physics_warning_lastinfo=np.zeros((substeps+1,8),np.int32),
        initial_integration=full,final_integration=end,control_integration_before=full[None],control_previous_action_before=previous[None],control_history_before=flat(history)[None],
        previous_action=previous[None],history=flat(history)[None],state=sensed[None],features=np.zeros((1,1000),np.float32),
        normalized_head=head[None],base_target=base[None],delta=delta[None],action=action[None],raw_proposal=raw[None],actual_normalized_action=actual[None],
        final_previous_action=action,final_recorded_controls=np.array(control+1),**{'final_history_'+k:v for k,v in after.items()})
    r=dict(segment='synthetic',requested_controls=1,attempted_controls=1,attempted_precontrols=1,physics_steps=substeps,controller_recorded_controls=control+1,
        completed_full_controls=int(substeps==10),full_segment_completed=substeps==10,failure=None if substeps==10 else {'global_control':control,'substep':substeps})
    cap=None
    if substeps<10:
        cap=dict(integration_at_failure=end,qpos_at_failure=q,qvel_at_failure=dq,previous_action_after=action,previous_action_before=previous,
            warning_counts_at_failure=np.zeros(8,np.int32),warning_lastinfo_at_failure=np.zeros(8,np.int32),integration_before=full,
            global_control=np.array(control),recorded_controls_after=np.array(control+1),head_input_features=np.zeros((1,1000),np.float32),head_output_0=head[None],
            **{'history_before_'+k:v for k,v in history.items()},**{'history_after_'+k:v for k,v in after.items()})
    return a,r,c,span,previous,history,cap

def run(data):
    a,r,c,span,prior,history,cap=data
    return segment(a,r,int(a['global_control'][0]),1,prior,history,lambda *args:np.zeros(1000,np.float32),terms,c,span,Checks(),cap)

class SavedChecks(unittest.TestCase):
    def test_phase_boundaries(self):self.assertEqual([phase(i) for i in (0,249,250,1268,1269,1569,1818)],[0,0,1,1,2,2,2])
    def test_repeated_clock(self):
        self.assertNotEqual(times(0.,1000)[-1],2.)
        self.assertEqual(times(times(0.,1000)[-1],1)[-1],times(0.,1001)[-1])
    def test_learned_clipped_inverse_prior(self):
        data=synthetic(250);prior,history,_=run(data)
        self.assertTrue(np.all(prior==3.));self.assertTrue(np.all(history['actions'][0]==0.))
        self.assertGreater(data[0]['raw_proposal'][0,0],1.)
    def test_unclipped_raw_terminal_prior(self):
        data=synthetic(1569);prior,_,_=run(data)
        self.assertTrue(np.all(prior==6.));self.assertTrue(np.all(data[0]['actual_normalized_action']==3.))
    def test_terminal_hold_history_carries_previous(self):
        first=synthetic(1569);prior,history,_=run(first)
        second=synthetic(1570,previous=prior,history=history);_,last,_=run(second)
        self.assertTrue(np.all(last['actions'][0]==6.));self.assertTrue(np.all(last['actions'][1]==0.))
    def test_partial_failure_capsule(self):run(synthetic(291,6))
    def test_bad_capsule_previous_rejected(self):
        data=synthetic(291,6);data[-1]['previous_action_after']=np.zeros(23,np.float32)
        with self.assertRaises(AssertionError):run(data)
    def test_head_cast_before_arithmetic(self):
        data=synthetic(250);data[0]['delta']=(data[3].astype(np.float32)*data[0]['normalized_head']).astype(np.float64)
        # span2 happens to be exact; choose a rounded non-power-of-two scale to distinguish.
        data=list(synthetic(250));data[3][0]=np.float32(2.1234567);a=data[0]
        correct=data[3]*a['normalized_head'][0].astype(np.float64)
        a['delta']=correct[None];a['raw_proposal']=(np.asarray(data[2]['default_q'])+correct)[None]
        a['delta'][0,0]=np.float64(np.float32(data[3][0])*a['normalized_head'][0,0])
        with self.assertRaises(AssertionError):run(data)
    def test_terminal_features_must_zero(self):
        data=synthetic(1269);data[0]['features'][0,0]=1.
        with self.assertRaises(AssertionError):run(data)
    def test_source_clock_hold_clamped(self):
        data=synthetic(1818);self.assertEqual(data[0]['source_frame'][0],1579);run(data)
        data[0]['source_frame'][0]=1829
        with self.assertRaises(AssertionError):run(data)
    def test_signed_zero_is_byte_distinct(self):
        with self.assertRaises(AssertionError):Checks().exact(np.array([-0.]),np.array([0.]),'signedzero')
    def test_consumed_path_normalization_and_conflict(self):
        contains({'E:/x/a':'a'},'/mnt/e/x/a','a')
        with self.assertRaises(AssertionError):contains({'E:/x/a':'a','/mnt/e/x/a':'b'},'E:/x/a','a')
    def test_full58_group_sum(self):
        rng=np.random.default_rng(3);gain=rng.normal(size=(23,58));tangent=rng.normal(size=58)
        groups,other,v,error=grouped(gain,tangent);self.assertEqual(groups.shape,(6,23));self.assertLess(error,1e-12)
        np.testing.assert_allclose(other+v,gain@tangent,rtol=1e-13,atol=1e-13)
    def test_nested_clips_full_gain(self):
        gain=np.ones((23,58));diff=lambda a,b:np.ones(58);limits=np.tile([-.05,.05],(23,1))
        target,raw,feedback,preclip=committed_target(diff,np.zeros(59),np.zeros(23),gain,np.zeros(30),np.zeros(29),limits)
        np.testing.assert_array_equal(raw,np.full(23,58.));np.testing.assert_array_equal(feedback,np.full(23,.1));np.testing.assert_array_equal(target,np.full(23,.05))
    def test_forbidden_model_native_imports(self):
        for name in ('saved_common.py','release_checks.py','audit_saved.py','fixed_maps.py','prepare_request.py'):
            tree=ast.parse((BASE/name).read_text())
            modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
            self.assertFalse(any(m.split('.')[0] in ('torch','onnxruntime','mujoco','direct_runtime','evaluation_gate') for m in modules))

if __name__=='__main__':unittest.main()
