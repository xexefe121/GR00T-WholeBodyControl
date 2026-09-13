"""Inference-only goal ablations at identical recorded pre-control robot inputs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
BASE=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
OUT=BASE/'bfm_goal_counterfactual_v1'
OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import PACKAGE,MODEL,corrected_goal,load_motion,load_case_motion
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMHistory,BFMZeroInference,load_contract,reference_features,state_and_terms


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}


def stats(value):
    return dict(p50=float(np.percentile(value,50)),p95=float(np.percentile(value,95)),max=float(np.max(value)),mean=float(np.mean(value)))


def joint_delta(value):
    return {group:dict(rmse_rad=float(np.sqrt(np.mean(value[:,span]**2))),
        absolute_component_rad=stats(np.abs(value[:,span])),per_control_l2_rad=stats(np.linalg.norm(value[:,span],axis=1)))
        for group,span in (('legs',slice(0,12)),('waist',slice(12,13)),('arms',slice(13,23)))}


torch.set_num_threads(1)
source_files=[Path(__file__),ROOT/'gear_sonic/scripts/evaluate_g1_true23_bfmzero.py',
              ROOT/'gear_sonic/utils/g1_true23_bfmzero_inference.py',ROOT/'gear_sonic/utils/g1_true23_step1b_mujoco.py']
source_hashes={str(p):sha(p) for p in source_files}
for p in source_files: (OUT/(p.stem+'_snapshot.py')).write_bytes(p.read_bytes())
contract_path=PACKAGE/'bfmzero_inspect_v1/config.yaml'
weights_path=PACKAGE/'bfmzero_inference_v1/inference.safetensors'
contract=load_contract(contract_path);policy=BFMZeroInference(weights_path)
native_path=ROOT.parent/'GR00T-WholeBodyControl'/MODEL
native=mujoco.MjModel.from_xml_path(str(native_path));limits=native.jnt_range[1:]
variant_names=['original','canonical_A','canonical_floor_B','retarget_v3','retarget_floor_v4']
folders={'canonical_A':'mjbatch_original_canonical_inputs_v1','canonical_floor_B':'mjbatch_original_canonical_floor_inputs_v1',
         'retarget_v3':'mjbatch_intent_inputs_v1','retarget_floor_v4':'mjbatch_intent_floor_inputs_v1'}
pairs=[('original','canonical_A'),('canonical_A','canonical_floor_B'),('original','canonical_floor_B'),
       ('canonical_A','retarget_v3'),('retarget_v3','retarget_floor_v4'),('original','retarget_v3'),('original','retarget_floor_v4')]
rows=[]
for clip in ('pico','walk002'):
    output=OUT/clip;output.mkdir()
    baseline=ART/f'bfm_{clip}_feedback_v2'
    report=json.loads((baseline/'report.json').read_text());trace=read_npz(baseline/'trace.npz')
    assert report['position_gain']==1 and report['yaw_gain']==2 and report['goal_horizon']==8 and report['arm_reference'] is False
    original,timeline,original_path=load_motion(clip)
    assert Path(report['reference_path'])==original_path
    phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
    assert len(trace['action'])>=phase['control_stop']
    controls=np.rint(np.linspace(phase['control_start'],phase['control_stop']-1,100)).astype(int)
    assert len(np.unique(controls))==100
    previous=np.concatenate((np.zeros((1,23),np.float32),trace['action'][:-1]),axis=0)
    history=BFMHistory();state_error=0.;history_error=0.
    for control in range(int(controls[-1])+1):
        q,dq=trace['qpos'][control],trace['qvel'][control]
        sensed,terms=state_and_terms(q[7:],dq[6:],q[3:7],dq[3:6],previous[control],contract['default_q'])
        hist=history.before_update(terms)
        state_error=max(state_error,float(np.max(np.abs(sensed-trace['state'][control]))))
        history_error=max(history_error,float(np.max(np.abs(hist-trace['history'][control]))))
    assert state_error==0 and history_error==0
    saved=dict(control=controls,source_frame=controls+11,pre_qpos=trace['qpos'][controls],pre_qvel=trace['qvel'][controls],
        actor_state=trace['state'][controls],actor_history=trace['history'][controls],previous_rescaled_action=previous[controls],
        baseline_saved_rescaled_action=trace['action'][controls],baseline_saved_clipped_target=trace['target'][controls])
    reference_hashes={};receipt_hashes={};features={}
    for name in variant_names:
        if name=='original': motion,path=original,original_path
        else:
            motion,other_timeline,path=load_case_motion(clip,BASE/folders[name]/clip/'reference.npz')
            assert other_timeline==timeline
            receipt_hashes[name]=sha(path.parent/'portable_receipt.json')
        reference_hashes[name]=dict(path=str(path),sha256=sha(path))
        state,privileged=reference_features(motion,contract)
        frames=controls[:,None]+11+np.arange(8)[None,:]
        features[name]=(state[frames],privileged[frames])
        latents=[];actions=[];raw_targets=[];clipped_targets=[]
        for control in controls:
            frame=int(control)+11
            z=corrected_goal(policy,state,privileged,motion,frame,trace['qpos'][control],8,1.,2.,'published-world-unscaled')
            raw=policy.actor(torch.from_numpy(trace['state'][control:control+1]),
                torch.from_numpy(previous[control:control+1]),torch.from_numpy(trace['history'][control:control+1]),z)[0].numpy()
            action=raw*5.
            target=contract['default_q']+action*.25*contract['training_effort']/contract['kp']
            latents.append(z[0].numpy());actions.append(action);raw_targets.append(target)
            clipped_targets.append(np.clip(target,limits[:,0],limits[:,1]))
        for key,value in (('latent',latents),('rescaled_action',actions),('raw_target',raw_targets),('clipped_target',clipped_targets)):
            saved[name+'_'+key]=np.asarray(value)
        print(json.dumps(dict(clip=clip,variant=name,source_samples=len(controls))),flush=True)
    parity_action=float(np.max(np.abs(saved['original_rescaled_action']-saved['baseline_saved_rescaled_action'])))
    parity_target=float(np.max(np.abs(saved['original_clipped_target']-saved['baseline_saved_clipped_target'])))
    assert parity_action<1e-5 and parity_target<1e-5,(parity_action,parity_target)
    comparisons=[]
    for left,right in pairs:
        z0,z1=saved[left+'_latent'],saved[right+'_latent']
        cosine=np.sum(z0*z1,axis=1)/(np.linalg.norm(z0,axis=1)*np.linalg.norm(z1,axis=1))
        cosine=np.clip(cosine,-1,1)
        s0,p0=features[left];s1,p1=features[right]
        feature_delta={}
        for field,span in (('height',slice(0,1)),('relative_body_position',slice(1,73)),('relative_body_rotation6',slice(73,223)),
                           ('world_heading_body_linear_velocity',slice(223,298)),('world_heading_body_angular_velocity',slice(298,373))):
            feature_delta[field]=stats(np.abs(p1[...,span]-p0[...,span]))
        for field,span in (('joint_position',slice(0,23)),('joint_velocity',slice(23,46)),('gravity',slice(46,49)),('world_root_gyro',slice(49,52))):
            feature_delta['goal_state_'+field]=stats(np.abs(s1[...,span]-s0[...,span]))
        comparisons.append(dict(left=left,right=right,latent_cosine=stats(cosine),latent_l2=stats(np.linalg.norm(z1-z0,axis=1)),
            rescaled_action_delta_rms_by_group={group:float(np.sqrt(np.mean((saved[right+'_rescaled_action'][:,span]-saved[left+'_rescaled_action'][:,span])**2)))
                for group,span in (('legs',slice(0,12)),('waist',slice(12,13)),('arms',slice(13,23)))},
            raw_target_delta=joint_delta(saved[right+'_raw_target']-saved[left+'_raw_target']),
            clipped_target_delta=joint_delta(saved[right+'_clipped_target']-saved[left+'_clipped_target']),
            reference_feature_component_absolute_delta=feature_delta))
    np.savez_compressed(output/'counterfactuals.npz',**saved)
    item=dict(clip=clip,source_requested_controls=phase['requested_controls'],source_phase=phase,
        sampled_controls=controls.tolist(),selection='100 unique evenly spaced source control indices, rounded to nearest integer; source endpoints included',
        precontrol_index_rule='qpos/control and qvel/control; state/control and history/control; previous action/control-1; requested source frame/control+11',
        reconstructed_full_prefix_state_max_abs_error=state_error,reconstructed_full_prefix_history_max_abs_error=history_error,
        original_action_parity_max_abs=parity_action,original_clipped_target_parity_max_abs_rad=parity_target,
        original_baseline_provenance_sidecar_present=(baseline/'provenance.json').exists(),
        baseline_trace_path=str(baseline/'trace.npz'),baseline_trace_sha256=sha(baseline/'trace.npz'),
        baseline_report_sha256=sha(baseline/'report.json'),reference_hashes=reference_hashes,portable_receipt_hashes=receipt_hashes,
        comparisons=comparisons,array_archive_sha256=sha(output/'counterfactuals.npz'),
        counterfactual_inference_only=True,physics_steps_executed=0,closed_loop_quality_claim=False,distribution_shift_proven=False)
    (output/'report.json').write_text(json.dumps(item,indent=2,allow_nan=False));rows.append(item)
assert all(sha(p)==h for p,h in source_hashes.items())
summary=dict(kind='BFM_goal_counterfactual_at_identical_recorded_measured_inputs',source_hashes=source_hashes,
    weights_path=str(weights_path),weights_sha256=sha(weights_path),config_sha256=sha(contract_path),native_model_sha256=sha(native_path),
    position_gain=1.,yaw_gain=2.,goal_horizon=8,goal_gyro_convention='published-world-unscaled',torch_threads=1,
    clips=[dict(clip=r['clip'],report_sha256=sha(OUT/r['clip']/'report.json'),original_action_parity_max_abs=r['original_action_parity_max_abs']) for r in rows],
    counterfactual_inference_only=True,physics_steps_executed=0,closed_loop_quality_claim=False,distribution_shift_proven=False,
    limitations=['The100 states are sampled from the original-goal closed-loop trajectory, not the other variants trajectories.',
                 'Large goal/action differences measure policy sensitivity; training-support or out-of-distribution status is not established.',
                 'Old baseline folders lack original provenance sidecars; inputs are hash-bound now and reproduced action parity is tested.',
                 'Central derivative variants need one additional future pose20ms beyond the8 goal samples; no real-time claim.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary['clips']),flush=True)
