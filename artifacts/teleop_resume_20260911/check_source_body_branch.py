"""Isolate current vs original-native body encoding on fixed expert states."""
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from gear_sonic.utils.g1_true23_direct_body_goal import DirectBodySinglePolicy,causal_body_features
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1')


def archive(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def main():
    output=BASE/'source_body_branch_diagnostic_v1';output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    bank=BASE/'focused_walk002_bank_v1';expert=archive(bank/'expert.npz')
    meta=json.loads((bank/'bank.json').read_text());row=meta['clips'][1]
    ids=np.flatnonzero((expert['resets'][:,0]==1)&(expert['resets'][:,1]>=row['source_start']+11)&(expert['resets'][:,1]<row['source_stop']+11))
    frames=expert['resets'][ids,1].astype(int);base=expert['features'][ids];targets=torch.tensor(expert['targets'][ids])
    current_encoded=archive(bank/'bfm_reference_inputs_v1.npz')
    x=np.concatenate((base,current_encoded['body_1'][frames],current_encoded['alpha_1'][frames,None]),-1).astype(np.float32)
    model,c,floor,original,timeline=load_case('walk002');data=mujoco.MjData(model)
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    native=archive(bundle/'walk002/native_original.npz')
    neutral={k:v[:1] for k,v in archive(bundle/'walk003/native_original.npz').items() if k!='fps'}
    body_ids=[model.body(name).id for name in c['body_names']]
    native_encoded=causal_body_features(native,c)
    hybrid_joint=floor['joint_pos'].copy();hybrid_joint[:,13:]=native['joint_pos'][:,13:]
    positions=[];quaternions=[];task_points={'floor':[],'original_native':[],'original_native_arms':[]}
    task_spec=meta['tasks']
    for frame in range(len(hybrid_joint)):
        q=np.r_[floor['body_pos_w'][frame,0],floor['body_quat_w'][frame,0],hybrid_joint[frame]]
        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        positions.append(data.xpos[body_ids].copy());quaternions.append(data.xquat[body_ids].copy())
    hybrid=dict(joint_pos=hybrid_joint,body_pos_w=np.asarray(positions),body_quat_w=np.asarray(quaternions))
    hybrid_encoded=causal_body_features(hybrid,c)
    for label,motion in [('floor',floor),('original_native',native),('original_native_arms',hybrid)]:
        for frame in frames:
            q=np.r_[motion['body_pos_w'][frame,0],motion['body_quat_w'][frame,0],motion['joint_pos'][frame]]
            data.qpos[:]=q;mujoco.mj_kinematics(model,data)
            points=np.stack([data.xpos[model.body(task['target_body']).id]+data.xmat[model.body(task['target_body']).id].reshape(3,3)@task['target_point'] for task in task_spec])
            actual=points-q[:3];desired=original['source_task_position_w'][frame]-original['source_qpos29'][frame,:3]
            task_points[label].append(np.linalg.norm(actual-desired,axis=-1))
    variants={'current_floor_body':x.copy()}
    for label,encoded in [('original_native_body_only',native_encoded),('original_native_arms_body_only',hybrid_encoded)]:
        changed=x.copy();changed[:,1323:1748]=encoded[frames];variants[label]=changed
    cases=[]
    weights=ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
    for model_name,path in [('initial',BASE/'range_normalized_initial_v1/actor_initial.pt'),('focused_v2_40',BASE/'focused_walk002_pilot_v2/actor_00040.pt')]:
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        actor=DirectBodySinglePolicy(weights,c,neutral,checkpoint['actor']['mean'].numpy(),checkpoint['actor']['scale'].numpy(),smooth_action_tau=.02,normalize_range_adapter=True)
        actor.load_state_dict(checkpoint['actor'])
        with torch.no_grad():
            for label,inputs in variants.items():
                prediction=actor.target(actor(torch.from_numpy(inputs)));error=prediction-targets
                case=dict(model=model_name,variant=label,joint_target_rmse_rad=float(error.square().mean().sqrt()),
                    leg_target_rmse_rad=float(error[:,:12].square().mean().sqrt()),arm_target_rmse_rad=float(error[:,13:].square().mean().sqrt()))
                cases.append(case);print(json.dumps(case),flush=True)
    report=dict(samples=len(frames),clip='walk002',all1323_task_state_history_features_unchanged=True,
        only_current_body425_encoding_changed=True,backward_derivatives_only=True,
        native_arm_vs_floor_max_joint_difference_rad=float(abs(native['joint_pos'][frames,13:]-floor['joint_pos'][frames,13:]).max()),
        native_arm_vs_floor_joint_rms_difference_rad=float(np.sqrt(np.mean((native['joint_pos'][frames,13:]-floor['joint_pos'][frames,13:])**2))),
        geometric_hand_head_relative_p95_m={key:np.percentile(value,95,axis=0).tolist() for key,value in task_points.items()},
        cases=cases,diagnostic_original_motion_lookup_not_live_controller=True,physical_rollouts_performed=False,simulation_qualified=False,hardware_authorized=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)


if __name__=='__main__':main()
