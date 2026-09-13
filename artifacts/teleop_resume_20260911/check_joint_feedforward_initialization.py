"""Fixed current-reference target comparisons; no dynamics or learned fitting."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1')


def main():
    output=BASE/'joint_feedforward_initialization_v1';output.mkdir(exist_ok=False)
    bank=BASE/'focused_walk002_bank_v1'
    metadata=json.loads((bank/'bank.json').read_text());row=metadata['clips'][1]
    with np.load(bank/'expert.npz') as z:
        selected=(z['resets'][:,0]==1)&(z['resets'][:,1]>=row['source_start']+11)&(z['resets'][:,1]<row['source_stop']+11)
        features=z['features'][selected];frames=z['resets'][selected,1].astype(int);teacher=z['targets'][selected]
    with np.load(bank/'bfm_reference_inputs_v1.npz') as z:
        inputs=np.concatenate((features,z['body_1'][frames],z['alpha_1'][frames,None]),-1).astype(np.float32)
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    c=json.loads((bundle/'contract.json').read_text());limits=np.asarray(c['joint_limits'])
    with np.load(bundle/'walk003/native_original.npz') as z:neutral=z['joint_pos'][0]
    actor=BASE/'range_normalized_initial_v1/actor_initial.onnx'
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(actor),sess_options=options,providers=['CPUExecutionProvider'])
    with np.load(actor.with_suffix('.normalization.npz')) as z:
        baseline=session.run(None,{'features':inputs})[0]*z['span']+z['default']
    baseline=np.clip(baseline,limits[:,0],limits[:,1])
    pose_delta=features[:,56:79]-neutral
    velocity_term=(np.asarray(c['kd'])/np.asarray(c['kp']))*features[:,79:102]
    cases=[]
    for pose_gain in (0.,.25,.5,1.):
        for velocity_gain in (0.,.25,.5,1.):
            raw=baseline+pose_gain*pose_delta+velocity_gain*velocity_term
            target=np.clip(raw,limits[:,0],limits[:,1]);error=target-teacher
            clipped=(raw<limits[:,0])|(raw>limits[:,1])
            report=dict(pose_gain=pose_gain,velocity_gain=velocity_gain,
                joint_target_rmse_rad=float(np.sqrt(np.mean(error**2))),
                leg_target_rmse_rad=float(np.sqrt(np.mean(error[:,:12]**2))),
                arm_target_rmse_rad=float(np.sqrt(np.mean(error[:,13:]**2))),
                waist_target_rmse_rad=float(np.sqrt(np.mean(error[:,12]**2))),
                native_target_clamp_fraction=float(clipped.mean()),clamped_states_fraction=float(clipped.any(-1).mean()),
                max_target_excess_before_clamp_rad=float(np.maximum(0,np.maximum(limits[:,0]-raw,raw-limits[:,1])).max()),
                target_change_max_rad=float(np.abs(target-baseline).max()))
            cases.append(report)
    report=dict(actor=str(actor),samples=len(frames),clip='walk002',source_start_frame=int(frames.min()),source_stop_frame_exclusive=int(frames.max()+1),
        formula='clip(baseline + pose_gain*(received_joint - native_neutral_joint) + velocity_gain*(kd/kp)*backward_received_joint_velocity,native_bounds)',
        all_23_joints=True,future_reference_frames=0,fit_performed=False,physical_rollout_performed=False,
        tracking_qualification=False,hardware_authorized=False,cases=cases)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/'fixed_targets.npz',baseline=baseline,teacher=teacher,pose_delta=pose_delta,velocity_term=velocity_term,source_frame=frames)
    for case in cases:print(json.dumps(case),flush=True)


if __name__=='__main__':main()
