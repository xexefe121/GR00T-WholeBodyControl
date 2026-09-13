"""Compare terminal encodings on identical measured successful standing states."""
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
    output=BASE/'terminal_body_goal_diagnostic_v1';output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    bank=BASE/'focused_walk002_bank_v1';data=archive(bank/'expert.npz')
    indices=archive(bank/'focused_quiet_rows.npz')['row_ids']
    features=data['features'][indices];targets=torch.tensor(data['targets'][indices]);rows=data['resets'][indices]
    bfm=archive(bank/'bfm_reference_inputs_v1.npz');frames=rows[:,1].astype(int)
    x=np.concatenate((features,bfm['body_1'][frames],bfm['alpha_1'][frames,None]),-1).astype(np.float32)
    model,c,motion,original,timeline=load_case('walk002');kinematics=mujoco.MjData(model)
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    native=archive(bundle/'walk002/native_original.npz')
    neutral={k:v[:1] for k,v in archive(bundle/'walk003/native_original.npz').items() if k!='fps'}
    ids=[model.body(name).id for name in c['body_names']]
    desired=original['source_task_position_w'][-1];task_spec=json.loads((bank/'bank.json').read_text())['tasks']
    def body(q):
        kinematics.qpos[:]=q;mujoco.mj_kinematics(model,kinematics)
        payload=dict(joint_pos=np.repeat(q[7:][None],2,0),body_pos_w=np.repeat(kinematics.xpos[ids][None],2,0),
            body_quat_w=np.repeat(kinematics.xquat[ids][None],2,0))
        points=np.stack([kinematics.xpos[model.body(task['target_body']).id]+kinematics.xmat[model.body(task['target_body']).id].reshape(3,3)@task['target_point'] for task in task_spec])
        mismatch=np.linalg.norm((points-q[:3])-(desired-original['source_qpos29'][-1,:3]),axis=-1)
        return causal_body_features(payload,c)[-1],mismatch
    current_q=np.r_[motion['body_pos_w'][-1,0],motion['body_quat_w'][-1,0],motion['joint_pos'][-1]]
    original_q=np.r_[native['body_pos_w'][-1,0],native['body_quat_w'][-1,0],native['joint_pos'][-1]]
    arms_q=current_q.copy();arms_q[20:]=original_q[20:]
    current,current_error=body(current_q);native_body,native_error=body(original_q);arms,arms_error=body(arms_q)
    variants={}
    variants['current_terminal']=x.copy()
    for name,encoded in [('original_native_body_only',native_body),('original_native_arms_body_only',arms)]:
        changed=x.copy();changed[:,1323:1748]=encoded;variants[name]=changed
    changed=x.copy();changed[:,1748]=0;variants['alpha_zero_diagnostic_only']=changed
    # Goal encoder data changes here; every original hand/head feature and
    # measured state/history remains untouched in these diagnostic variants.
    reports=[]
    weights=ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
    for model_name,path in [('initial',BASE/'range_normalized_initial_v1/actor_initial.pt'),('focused_v2_40',BASE/'focused_walk002_pilot_v2/actor_00040.pt')]:
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        actor=DirectBodySinglePolicy(weights,c,neutral,checkpoint['actor']['mean'].numpy(),checkpoint['actor']['scale'].numpy(),
            smooth_action_tau=.02,normalize_range_adapter=True)
        actor.load_state_dict(checkpoint['actor'])
        with torch.no_grad():
            for name,inputs in variants.items():
                prediction=actor.target(actor(torch.from_numpy(inputs)));error=prediction-targets
                report=dict(model=model_name,variant=name,all_target_rmse_rad=float(error.square().mean().sqrt()),
                    leg_target_rmse_rad=float(error[:,:12].square().mean().sqrt()),arm_target_rmse_rad=float(error[:,13:].square().mean().sqrt()),
                    continuous_hold_target_rmse_rad=float(error[300:].square().mean().sqrt()),
                    target_to_actual_pose_rms_rad=float((prediction-torch.tensor(rows[:,9:32])).square().mean().sqrt()))
                reports.append(report);print(json.dumps(report),flush=True)
    report=dict(samples=len(indices),same_actual_successful_quiet_states=True,actor_goal_blend_values=np.unique(x[:,-1]).tolist(),
        upper_joint_branch_max_change_rad=float(np.abs(current_q[20:]-arms_q[20:]).max()),
        leg_branch_max_change_rad=float(np.abs(current_q[7:19]-original_q[7:19]).max()),
        static_goal_task_relative_errors_m=dict(current=current_error.tolist(),original_native=native_error.tolist(),original_native_arms=arms_error.tolist()),
        cases=reports,physical_rollouts_performed=False,simulation_qualified=False,hardware_authorized=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)


if __name__=='__main__':main()
