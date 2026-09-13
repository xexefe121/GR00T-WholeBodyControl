"""Stub and saved-array tests only: no native steps, models or training."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import unittest
import numpy as np

BASE=Path(__file__).parent;SOURCE=BASE/'source_draft_v1'
sys.path.insert(0,str(SOURCE))
from clipped_feedback import clipped_component_feedback
from feedback_parity import FeedbackParity,CONTROL_FIELDS,SAMPLE_FIELDS,STEP_FIELDS
from proposal_evidence import trace_arrays,exact

CONTRACT=json.loads(Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text())
with np.load(BASE.parent/'one_step_physical_student_evaluation_v1/nominal/trace.npz') as archive:
    ORIGINAL={key:archive[key].copy() for key in archive.files}


def load_node(path,name,namespace):
    tree=ast.parse(path.read_text());node=next(node for node in tree.body if getattr(node,'name',None)==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


History=load_node(SOURCE/'gear_sonic/utils/g1_true23_bfm_seed_observations.py','BFMHistory',dict(np=np))


def flat_history(history):return np.concatenate([history.data[k].reshape(-1) for k in sorted(history.data)])
def terms(state,previous):return dict(actions=previous,base_ang_vel=state[49:52],dof_pos=state[:23],dof_vel=state[23:46],projected_gravity=state[46:49])
def all_exact(pairs):return all(exact(a,b) for _,a,b in pairs)


class FeedbackTests(unittest.TestCase):
    def test_actual_first_clip_replaces_only_clipped_components(self):
        row=264;raw=ORIGINAL['action'][row].copy();snapshot=raw.copy()
        feedback,native,mask=clipped_component_feedback(raw,ORIGINAL['raw_proposal'][row],ORIGINAL['target'][row],CONTRACT,True)
        actual=ORIGINAL['actual_normalized_action'][row]
        self.assertTrue(mask.any());self.assertTrue(exact(mask,native))
        self.assertTrue(exact(feedback[mask],actual[mask]));self.assertTrue(exact(feedback[~mask],raw[~mask]))
        self.assertTrue(exact(raw,snapshot));self.assertFalse(exact(feedback,raw))

    def test_unclipped_signed_zero_and_roundtrip_differences_are_preserved(self):
        combined=np.linspace(-.123,.321,23,dtype=np.float32);combined[7]=np.float32(-0.)
        target=np.asarray(CONTRACT['default_q'])+.01
        feedback,_,mask=clipped_component_feedback(combined,target,target.copy(),CONTRACT,True)
        self.assertFalse(mask.any());self.assertTrue(exact(feedback,combined));self.assertTrue(np.signbit(feedback[7]))
        actual=((target-np.asarray(CONTRACT['default_q']))*np.asarray(CONTRACT['kp'])/(.25*np.asarray(CONTRACT['training_effort']))).astype(np.float32)
        self.assertFalse(exact(actual,combined))

    def test_upper_lower_clips_and_exact_boundary(self):
        limits=np.asarray(CONTRACT['joint_limits']);rawpos=np.asarray(CONTRACT['default_q']).copy()
        rawpos[5]=limits[5,1]+.2;rawpos[11]=limits[11,0]-.3;rawpos[3]=limits[3,1]
        target=np.clip(rawpos,limits[:,0],limits[:,1]);combined=np.linspace(-2,2,23,dtype=np.float32)
        feedback,native,mask=clipped_component_feedback(combined,rawpos,target,CONTRACT,True)
        self.assertEqual(np.flatnonzero(mask).tolist(),[5,11]);self.assertTrue(exact(native,mask))
        actual=((target-np.asarray(CONTRACT['default_q']))*np.asarray(CONTRACT['kp'])/(.25*np.asarray(CONTRACT['training_effort']))).astype(np.float32)
        self.assertTrue(exact(feedback[mask],actual[mask]));self.assertTrue(exact(feedback[~mask],combined[~mask]))

    def test_disabled_startup_and_terminal_retain_raw_even_when_clipped(self):
        for row in (4,264):
            feedback,native,mask=clipped_component_feedback(ORIGINAL['action'][row],ORIGINAL['raw_proposal'][row],ORIGINAL['target'][row],CONTRACT,False)
            self.assertTrue(native.any());self.assertFalse(mask.any());self.assertTrue(exact(feedback,ORIGINAL['action'][row]))

    def test_invalid_output_fails_before_feedback_commit(self):
        values=np.zeros(23,np.float32);raw=np.zeros(23);target=raw.copy()
        for bad in (np.zeros(23,np.float64),np.zeros(22,np.float32),np.full(23,np.nan,np.float32)):
            with self.assertRaises(ValueError):clipped_component_feedback(bad,raw,target,CONTRACT,True)
        self.assertTrue(exact(values,np.zeros(23,np.float32)))

    def test_actual_propose_method_with_stub_sessions(self):
        counts=dict(base=0,head=0)
        raw=np.linspace(-.02,.02,23,dtype=np.float32)
        base=np.asarray(CONTRACT['default_q'])+raw*.25*np.asarray(CONTRACT['training_effort'])/np.asarray(CONTRACT['kp'])
        delta=np.zeros(23,np.float32);delta[5]=1.;delta[11]=-1.
        def infer(seed,qpos,qvel,previous,history,frame,terminal=False):
            counts['base']+=1;return raw.copy(),base.copy(),np.zeros(52,np.float32)
        Runtime=load_node(SOURCE/'student_linear_runtime.py','LinearStudentRuntime',dict(np=np,time=time,infer_base=infer,FEATURES=1069,clipped_component_feedback=clipped_component_feedback))
        class Head:
            def run(self,*args):counts['head']+=1;return [delta[None].copy()]
        for terminal,disabled in ((False,False),(False,True),(True,False)):
            obj=Runtime.__new__(Runtime);obj.c=CONTRACT;obj.limits=np.asarray(CONTRACT['joint_limits']);obj.head=Head()
            obj.features=lambda *args:np.zeros(1069,np.float32)
            seed=SimpleNamespace(previous_action=np.zeros(23,np.float32),recorded_controls=264,history=History())
            seed._terms=lambda q,dq,prior:(np.zeros(52,np.float32),terms(np.zeros(52,np.float32),prior))
            obj.seed=seed
            result=obj.propose(264,np.zeros(30),np.zeros(29),terminal=terminal,disable_head=disabled)
            used=np.zeros(23,np.float32) if terminal or disabled else delta
            combined=(raw+used*np.asarray(CONTRACT['kp'])/(.25*np.asarray(CONTRACT['training_effort']))).astype(np.float32)
            self.assertTrue(exact(result['action'],combined));self.assertTrue(exact(result['raw_combined_action'],combined))
            self.assertTrue(exact(seed.previous_action,result['feedback_action']));self.assertEqual(seed.recorded_controls,265)
            if terminal or disabled:self.assertTrue(exact(seed.previous_action,combined));self.assertFalse(result['feedback_clip_mask'].any())
            else:self.assertEqual(np.flatnonzero(result['feedback_clip_mask']).tolist(),[5,11])
        self.assertEqual(counts,dict(base=3,head=1))

    def test_real_history_has_no_early_actor_lag_effect(self):
        parity=FeedbackParity(ORIGINAL,CONTRACT);history=History();start=0
        for key in sorted(history.data):
            n=history.data[key].size;history.data[key][:]=ORIGINAL['control_history_before'][264,start:start+n].reshape(history.data[key].shape);start+=n
        h264=history.before_update(terms(ORIGINAL['state'][264],ORIGINAL['previous_action'][264]))
        self.assertTrue(exact(h264,ORIGINAL['history'][264]))
        h265=history.before_update(terms(ORIGINAL['state'][265],parity.first_feedback))
        self.assertTrue(exact(h265,ORIGINAL['history'][265]))
        self.assertTrue(exact(flat_history(history),parity.lag266))
        h266=history.before_update(terms(np.zeros(52,np.float32),np.ones(23,np.float32)))
        self.assertTrue(exact(h266,parity.lag266));self.assertTrue(exact(h266[:23],parity.first_feedback))

    def test_saved_prefix_and_transition_gates(self):
        parity=FeedbackParity(ORIGINAL,CONTRACT)
        trace={k:ORIGINAL[k][:265].copy() for k in CONTROL_FIELDS}
        trace.update({k:ORIGINAL[k][:266].copy() for k in ('qpos','qvel')})
        trace.update({k:ORIGINAL[k][:2651].copy() for k in SAMPLE_FIELDS})
        trace.update({k:ORIGINAL[k][:2650].copy() for k in STEP_FIELDS})
        feedback=[];masks=[]
        for row in range(265):
            f,_,m=clipped_component_feedback(ORIGINAL['action'][row],ORIGINAL['raw_proposal'][row],ORIGINAL['target'][row],CONTRACT,row>=250);feedback.append(f);masks.append(m)
        trace.update(raw_combined_action=ORIGINAL['action'][:265].copy(),feedback_action=np.asarray(feedback),feedback_clip_mask=np.asarray(masks))
        args=(265,trace,ORIGINAL['control_integration_before'][265],ORIGINAL['qpos'][265],ORIGINAL['qvel'][265],ORIGINAL['control_history_before'][265],parity.first_feedback,265,ORIGINAL['physics_warning_counts'][2650],ORIGINAL['physics_warning_lastinfo'][2650],ORIGINAL['physics_expected_time'][2650])
        self.assertTrue(all_exact(parity.before(*args)[1]))
        trace['physics_qpos'][21,0]=np.nextafter(trace['physics_qpos'][21,0],np.inf)
        self.assertFalse(all_exact(parity.before(*args)[1]))
        proposed=dict(previous_action=parity.first_feedback,history=ORIGINAL['history'][265],state=ORIGINAL['state'][265])
        self.assertTrue(all_exact(parity.after(265,proposed)[1]))
        self.assertTrue(all_exact(parity.after(266,dict(history=parity.lag266))[1]))
        wrong=parity.lag266.copy();wrong[0]=np.nextafter(wrong[0],np.float32(np.inf))
        self.assertFalse(all_exact(parity.after(266,dict(history=wrong))[1]))

    def test_empty_failure_trace_preserves_new_shapes(self):
        new_trace=load_node(SOURCE/'evaluate_physical_response_student.py','new_trace',dict(np=np))
        data=SimpleNamespace(qpos=np.zeros(30),qvel=np.zeros(29),time=0.,warning=SimpleNamespace(number=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32)))
        trace=new_trace(data)
        for key in ('raw_proposal','actual_normalized_action','raw_combined_action','feedback_action','native_target_clip_mask','feedback_clip_mask'):trace[key]=[]
        arrays=trace_arrays(trace)
        for key in ('raw_combined_action','feedback_action'):
            self.assertEqual(arrays[key].shape,(0,23));self.assertEqual(arrays[key].dtype,np.float32)
        for key in ('native_target_clip_mask','feedback_clip_mask'):
            self.assertEqual(arrays[key].shape,(0,23));self.assertEqual(arrays[key].dtype,np.bool_)


if __name__=='__main__':unittest.main(verbosity=2)
