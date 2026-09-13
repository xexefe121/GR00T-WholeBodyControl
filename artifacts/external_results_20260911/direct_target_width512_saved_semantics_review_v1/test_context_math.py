"""Synthetic causal timing, signed bytes, rejection and map-coverage regressions."""
import copy
import unittest
from saved_common import *
from context_math import context,features,nominal_features,map_selection
from audit_saved import segment
from test_saved import synthetic,run,terms
from release_checks import binding_identity


def distinct_history():
    h=history_zero()
    for i,name in enumerate(sorted(h)):
        h[name][:]=np.arange(h[name].size,dtype=np.float32).reshape(h[name].shape)+i*100
    return h


def rejected():
    a,r,c,span,prior,h,_=synthetic(250)
    after={k:a['final_history_'+k].copy() for k in h}
    outgoing=a['final_previous_action'].copy()
    final=a['final_integration'].copy()
    a['control_integration_before']=np.concatenate((a['control_integration_before'],final[None]))
    a['control_previous_action_before']=np.concatenate((a['control_previous_action_before'],outgoing[None]))
    a['control_history_before']=np.concatenate((a['control_history_before'],flat(after)[None]))
    r.update(requested_controls=2,attempted_precontrols=2,full_segment_completed=False,failure={'global_control':251,'reason':'head_fault'})
    cap=dict(integration_at_failure=final,qpos_at_failure=a['qpos'][-1].copy(),qvel_at_failure=a['qvel'][-1].copy(),
        previous_action_after=outgoing.copy(),previous_action_before=outgoing.copy(),
        warning_counts_at_failure=a['physics_warning_counts'][-1],warning_lastinfo_at_failure=a['physics_warning_lastinfo'][-1],
        integration_before=final,global_control=np.asarray(251),recorded_controls_after=np.asarray(251),
        head_input_features=features(np.zeros(1000,np.float32),outgoing,after,251)[None],
        **{'history_before_'+k:v.copy() for k,v in after.items()},**{'history_after_'+k:v.copy() for k,v in after.items()})
    return a,r,c,span,prior,h,cap


def check_rejected(data):
    a,r,c,span,prior,h,cap=data
    return segment(a,r,250,2,prior,h,lambda *args:np.zeros(1000,np.float32),terms,c,span,Checks(),cap)


class ContextChecks(unittest.TestCase):
    def test_exact_pre_update_layout(self):
        prior=np.arange(23,dtype=np.float32)+700;h=distinct_history();current=np.arange(1000,dtype=np.float32)
        x=features(current,prior,h,250)
        Checks().exact(x,np.concatenate((current,prior,*[h[k].reshape(-1) for k in sorted(h)])),'public layout')
        self.assertEqual(x.shape,(1323,));self.assertEqual(x.dtype,np.float32)

    def test_result_owns_bytes_and_does_not_shift_history(self):
        h=distinct_history();original=copy.deepcopy(h);p=np.ones(23,np.float32)
        result=context(p,h);result[:]=-999
        for k in h:Checks().exact(h[k],original[k],'not shifted')
        self.assertTrue(np.all(p==1))

    def test_signed_zero_preserved(self):
        h=history_zero();p=np.zeros(23,np.float32);p[0]=-0.;h['dof_pos'][0,0]=-0.
        x=context(p,h)
        self.assertTrue(np.signbit(x[0]));self.assertEqual(x.tobytes(),np.concatenate((p,flat(h))).tobytes())

    def test_current_history_is_not_staged(self):
        prior=np.arange(23,dtype=np.float32)+1;h=distinct_history();data=synthetic(250,previous=prior,history=h)
        run(data)
        staged=copy.deepcopy(h);_,values=terms(None,None,None,None,prior,None);advance(staged,values)
        data[0]['features'][0,1023:]=flat(staged)
        with self.assertRaises(AssertionError):run(data)

    def test_outgoing_prior_cannot_replace_incoming(self):
        data=synthetic(250);data[0]['features'][0,1000:1023]=data[0]['action'][0]
        with self.assertRaises(AssertionError):run(data)

    def test_current_goal_columns_checked(self):
        data=synthetic(250);data[0]['features'][0,999]=1
        with self.assertRaises(AssertionError):run(data)

    def test_causal_not_mean_blinded(self):
        data=synthetic(250,previous=np.ones(23,np.float32));data[0]['features'][0,1000:]=0
        with self.assertRaises(AssertionError):run(data)

    def test_full_public_dtype_and_width(self):
        for change in (lambda a:a.astype(np.float64),lambda a:a[:,:1000]):
            data=synthetic(250);data[0]['features']=change(data[0]['features'])
            with self.assertRaises(AssertionError):run(data)

    def test_applied_inverse_then_next_history(self):
        first=synthetic(250);previous,h,_=run(first)
        second=synthetic(251,previous=previous,history=h);prior,last,_=run(second)
        Checks().exact(second[0]['features'][0,1000:1023],previous,'next prior')
        Checks().exact(second[0]['features'][0,1023:],flat(h),'next incoming H')
        Checks().exact(last['actions'][0],previous,'new history oldest ownership')
        self.assertTrue(np.all(prior==3))

    def test_terminal_features_zero_but_incoming_history_retained(self):
        data=synthetic(1269,previous=np.ones(23,np.float32),history=distinct_history());prior,h,_=run(data)
        self.assertTrue(np.all(data[0]['features']==0));self.assertTrue(np.all(h['actions'][0]==1));self.assertTrue(np.all(prior==6))

    def test_nonfinite_context_rejected(self):
        h=history_zero();h['actions'][0,0]=np.nan
        with self.assertRaises(AssertionError):context(np.zeros(23,np.float32),h)

    def test_context_shape_dtype_and_missing_name(self):
        for mutation in ('dtype','shape','missing'):
            h=history_zero()
            if mutation=='dtype':h['actions']=h['actions'].astype(np.float64)
            if mutation=='shape':h['actions']=h['actions'][:3]
            if mutation=='missing':del h['actions']
            with self.assertRaises(AssertionError):context(np.zeros(23,np.float32),h)

    def test_nominal_query_columns_exclude_old_nuisance(self):
        x=np.arange(2*1069,dtype=np.float32).reshape(2,1069);p=np.ones((2,23),np.float32);h=np.ones((2,300),np.float32)*2
        a=nominal_features(x,p,h);changed=x.copy();changed[:,52:75]=-10;changed[:,1023:]=-20
        Checks().exact(a,nominal_features(changed,p,h),'reduced current columns')
        Checks().exact(a[:,1000:],np.concatenate((p,h),axis=1),'nominal true context')

    def test_rejected_proposal_retains_incoming_context(self):check_rejected(rejected())

    def test_rejected_context_corruption(self):
        data=rejected();data[-1]['head_input_features'][0,1000]=0
        with self.assertRaises(AssertionError):check_rejected(data)

    def test_rejected_history_not_committed(self):
        data=rejected();data[-1]['history_after_actions'][0]=3
        with self.assertRaises(AssertionError):check_rejected(data)

    def test_staged_capsule_history_checked(self):
        data=list(synthetic(291,6));data[-1]['staged_history_actions']=np.ones((4,23),np.float32)
        with self.assertRaises(AssertionError):run(data)

    def test_zero_returned_steps_still_has_issued_command(self):
        data=synthetic(291,0);run(data)
        self.assertEqual(len(data[0]['target']),1);self.assertEqual(len(data[0]['physics_torque']),0)

    def test_negative_or_over_budget_substeps_rejected(self):
        for value in (-1,11):
            data=synthetic(291,6);data[0]['physics_substeps'][0]=value
            with self.assertRaises(AssertionError):run(data)

    def test_causal_release_identity(self):
        b=dict(context_condition='causal',architecture=[1323,512,512,23],ordinary_final_step=81000,
            release_kind='ordinary81000_width512_warm_balanced_context_same_weight_fp64_export')
        binding_identity(b,Checks())
        for key,value in [('context_condition','blinded'),('architecture',[1000,256,256,23]),
                          ('ordinary_final_step',65000),('release_kind','ordinary65000_full58_same_weight_fp64_export')]:
            wrong=dict(b);wrong[key]=value
            with self.assertRaises(AssertionError):binding_identity(wrong,Checks())

    def test_matching_unique_map_coverage(self):
        c=dict(dataset=np.array([2,2]),control=np.array([250,252]),plan_control=np.array([250,250]),plan_local=np.array([0,2]),
            gain=np.zeros((2,23,58)),planned_state=np.zeros((2,59)),planned_target=np.zeros((2,23)),qpos=np.zeros((2,30)),qvel=np.zeros((2,29)),expert_target=np.zeros((2,23)))
        chosen,missing=map_selection(c,np.array([250,251,252]));self.assertEqual(chosen.tolist(),[250,252]);self.assertEqual(missing,[251])
        c['control'][1]=250
        with self.assertRaises(AssertionError):map_selection(c,np.array([250]))


if __name__=='__main__':unittest.main()
