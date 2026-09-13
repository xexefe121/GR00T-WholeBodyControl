"""Received-only task closure checks; no physical rollout or model fitting."""
from collections import deque
import argparse,copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from gear_sonic.utils.g1_true23_direct_body_goal import ReceivedBodyGoal,task_closure_signature,task_closure_standing
from gear_sonic.utils.g1_true23_received_features import prepare_reference,features_numpy

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=BASE/'task_closure_checks_v2')
    output=parser.parse_args().output;output.mkdir(exist_ok=False)
    reports=[]
    for clip in ('walk003','walk002','pico','walk008'):
        model,c,motion,original,timeline=load_case(clip)
        samples=[{k:v[i] for k,v in motion.items() if k!='fps'} for i in range(len(motion['joint_pos']))]
        tasks=list(zip(original['source_task_position_w'],original['source_task_quaternion_wxyz']))
        anchor=task_closure_signature(samples[0],*tasks[0],c)
        refs=prepare_reference(motion,original,c)
        alpha=0.;old_alpha=0.;closures=[];actual_alpha=[];old_alpha_trace=[];nonneutral=[]
        for frame,sample in enumerate(samples):
            signature=task_closure_signature(sample,*tasks[frame],c)
            velocities=dict(joint_velocity=refs['joint_velocity'][frame],root_velocity=refs['root_velocity'][frame],
                root_omega=refs['root_omega'][frame],feet_velocity=refs['feet_velocity'][frame],
                task_velocity=refs['task_velocity'][frame],task_omega=refs['task_omega'][frame])
            closed=task_closure_standing(signature,anchor,**velocities)
            near=min(np.max(abs(sample['joint_pos']-samples[0]['joint_pos'])),np.max(abs(sample['joint_pos']-c['default_q'])))<=.04
            moving=(not near) or np.max(abs(velocities['joint_velocity']))>.10 or np.linalg.norm(velocities['root_velocity'])>.025 or np.linalg.norm(velocities['root_omega'])>.10
            actual_alpha.append(alpha);old_alpha_trace.append(old_alpha);closures.append(closed)
            alpha+=np.clip(float(not closed)-alpha,-.04,.04)
            old_alpha+=np.clip(float(moving)-old_alpha,-.04,.04)
            if np.linalg.norm(signature['tasks']-anchor['tasks'],axis=-1).max()>.05:nonneutral.append(frame)
        assert max(actual_alpha[-25:])==0,(clip,actual_alpha[-1],old_alpha_trace[-1])
        assert nonneutral
        selected=nonneutral[len(nonneutral)//2]
        gesture=task_closure_signature(samples[selected],*tasks[selected],c)
        zero=dict(joint_velocity=np.zeros(23),root_velocity=np.zeros(3),root_omega=np.zeros(3),
            feet_velocity=np.zeros((2,3)),task_velocity=np.zeros((3,3)),task_omega=np.zeros((3,3)))
        assert not task_closure_standing(gesture,anchor,**zero)
        rotated=copy.deepcopy(anchor);angle=.2
        rx=np.array([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],[0.,np.sin(angle),np.cos(angle)]])
        rotated['task_rotation'][0]=rx@rotated['task_rotation'][0]
        assert not task_closure_standing(rotated,anchor,**zero)
        moved_foot=copy.deepcopy(anchor);moved_foot['feet'][0,0]+=.02
        assert not task_closure_standing(moved_foot,anchor,**zero)
        moving_zero=copy.deepcopy(zero);moving_zero['task_velocity'][0,0]=.04
        assert not task_closure_standing(anchor,anchor,**moving_zero)
        # Exercise the actual runtime adapter on the same owned terminal data.
        receiver=SimpleNamespace(gate=SimpleNamespace(epoch=0,fault=None),samples=deque([samples[0],samples[-1],samples[-1]]),tasks=deque([tasks[0],tasks[-1],tasks[-1]]))
        q=np.r_[sample['body_pos_w'][0],sample['body_quat_w'][0],sample['joint_pos']]
        base=features_numpy(q,np.zeros(29),refs,len(samples)-1,c['default_q'],np.zeros(23),np.zeros(300))
        current=ReceivedBodyGoal(c,task_closure_stand=True);old=ReceivedBodyGoal(c)
        fast=ReceivedBodyGoal(c,task_closure_stand=True)
        current.features(base,receiver);old.features(base,receiver);fast.advance_blend(base,receiver)
        current.alpha=1.;old.alpha=1.;fast.alpha=1.
        for _ in range(26):
            new_features=current.features(base,receiver);old_features=old.features(base,receiver)
            np.testing.assert_array_equal(new_features[:-1],old_features[:-1])
            assert new_features[-1]==np.float32(fast.advance_blend(base,receiver))
        assert new_features[-1]==0
        # A task-only stationary orientation gesture with identical robot-body
        # packets must also reactivate motion instead of being ignored.
        from scipy.spatial.transform import Rotation
        changed_quaternion=tasks[-1][1].copy()
        rotation=Rotation.from_quat(changed_quaternion[0,[1,2,3,0]])
        changed_quaternion[0]=(Rotation.from_rotvec([.2,0,0])*rotation).as_quat()[[3,0,1,2]]
        receiver.tasks[-1]=(tasks[-1][0],changed_quaternion)
        current.features(base,receiver)
        fast.advance_blend(base,receiver)
        assert current.alpha==fast.alpha
        assert current.alpha>0
        receiver.tasks[-1]=tasks[-1]
        # Unreceived suffix mutation cannot alter a current signature/predicate.
        future=copy.deepcopy(samples)
        for later in future[selected+1:]:later['joint_pos']=np.full(23,999.)
        a=task_closure_signature(samples[selected],*tasks[selected],c)
        b=task_closure_signature(future[selected],*tasks[selected],c)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        report=dict(clip=clip,frames=len(samples),terminal_alpha=float(new_features[-1]),old_terminal_alpha=float(old_features[-1]),
            closure_frames=int(sum(closures)),nonneutral_static_arm_stays_active=True,orientation_only_gesture_stays_active=True,
            nonneutral_foot_stays_active=True,moving_task_stays_active=True,all_other1748features_unchanged=True,future_mutation_no_effect=True)
        report['blend_without_bfm_encoding_matches']=True
        reports.append(report);print(json.dumps(report),flush=True)
    (output/'report.json').write_text(json.dumps(dict(cases=reports,passed=True,physical_rollouts_performed=False,
        simulation_qualified=False,hardware_authorized=False),indent=2)+'\n')


if __name__=='__main__':main()
