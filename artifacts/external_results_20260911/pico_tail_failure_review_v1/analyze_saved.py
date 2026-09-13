"""Read-only NumPy analysis of frozen PICO evidence. No model or policy imports."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
NEW = OUT.parent
OLD = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
RUN = NEW / 'pico_full_hard_restoration_v1'
BUNDLE = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
tr = np.load(RUN / 'trace.npz')
plans = json.loads((RUN / 'plans.json').read_text())
report = json.loads((RUN / 'report.json').read_text())
contract = json.loads((BUNDLE / 'contract.json').read_text())
orig = np.load(BUNDLE / 'pico/original29.npz')['source_qpos29']
motion_path = OLD / 'mjbatch_intent_floor_inputs_v1/pico/reference.npz'
motion = np.load(motion_path)
warm_path = RUN / 'restoration_03800_guided.npz'
warm = np.load(warm_path)['original_warm_targets']
names = contract['joint_names']
limits = np.asarray(contract['joint_limits'])

def stats(x):
    x = np.asarray(x)
    return dict(p50=float(np.percentile(x,50)), p95=float(np.percentile(x,95)), maximum=float(np.max(x)))

def yaw(q):
    w,x,y,z=q.T
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))

frame=tr['source_frame']
q=tr['qpos'][1:]
root_original=np.linalg.norm(q[:,:3]-orig[frame,:3],axis=1)
yaw_original=np.abs((yaw(q[:,3:7])-yaw(orig[frame,3:7])+np.pi)%(2*np.pi)-np.pi)*180/np.pi
root_declared=np.linalg.norm(tr['root_error'],axis=1)
legs=np.sqrt(np.mean(tr['joint_error'][:,:12]**2,axis=1))
arms=np.sqrt(np.mean(tr['joint_error'][:,13:]**2,axis=1))
tilt=np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1))
margin=np.minimum(q[:,7:]-limits[:,0],limits[:,1]-q[:,7:])
clip=np.abs(tr['feedback_correction_raw']) > .1

def segment(a,b):
    s=slice(a,b)
    return dict(control_start=a,control_stop_exclusive=b,source_seconds=[(a-350)*.02,(b-350)*.02],
        original_world_root_error_m=stats(root_original[s]),original_heading_error_deg=stats(yaw_original[s]),
        declared_root_error_m=stats(root_declared[s]),declared_leg_per_control_rmse_rad=stats(legs[s]),
        declared_arm_per_control_rmse_rad=stats(arms[s]),
        declared_ankle_origin_world_error_p95_m=np.percentile(tr['body_position_error'][s,2:4],95,axis=0).tolist(),
        actual_root_height_min_max_m=[float(q[s,2].min()),float(q[s,2].max())],
        actual_root_linear_speed_m_s=stats(np.linalg.norm(tr['qvel'][1:][s,:3],axis=1)),
        actual_tilt_rad=stats(tilt[s]),minimum_control_endpoint_joint_margin_rad=float(margin[s].min()),
        actual_physics_speed_ratio_peak=float(tr['velocity_ratio'][s].max()),
        feedback_raw_max_rad=float(np.abs(tr['feedback_correction_raw'][s]).max()),
        feedback_components_clipped_fraction=float(clip[s].mean()),
        feedback_controls_with_clip=int(clip[s].any(axis=1).sum()),
        planned_precontrol_joint_q_error_max_rad=float(np.abs(tr['qpos'][a:b,7:]-tr['planned_state'][s,7:30]).max()),
        planned_precontrol_root_error_max_m=float(np.linalg.norm(tr['qpos'][a:b,:3]-tr['planned_state'][s,:3],axis=1).max()))

def first_violation(witness):
    return witness['first_violation'][0] if witness is not None else None

def compact_plan(p, executed=True):
    seeds=p['seed_feasibility']
    sf=p.get('solver_feasibility')
    accepted=[] if not sf else [x for x in sf.get('iterations',[]) if x.get('accepted')]
    selected_ok=None
    if sf and sf.get('initial_rollout'):
        selected_ok=bool(sf['initial_rollout']['feasible'][0])
    if accepted:
        last=accepted[-1]
        selected_ok=bool(last['rollout_feasibility']['feasible'][last['selected_lane']])
    return dict(control=p['control'],source_seconds=(p['control']-350)*.02,
        cost=p.get('cost'),selected_seed=p['seed_selection']['selected'],
        seed_first_violations={k:first_violation(v) for k,v in seeds.items()},
        final_accepted_nominal_horizon_feasible=selected_ok,
        solver_status=None if not sf else sf.get('status'),
        imminent_checks=len(p.get('imminent_control_checks',[])),
        all_imminent_native_checks_pass=all(x['feasible'] for x in p.get('imminent_control_checks',[])) if executed else None)

recent=[compact_plan(p) for p in plans if 3700<=p['control']<3800]
failed=compact_plan(report['failure'],False)
shifts=[p['seed_first_violations']['shifted_mpc'] for p in recent+[failed]]
shifts=[v for v in shifts if v is not None]
idx=np.arange(3836,3841)
feed=np.clip(motion['joint_pos'][idx]+np.asarray(contract['kd'])/np.asarray(contract['kp'])*motion['joint_vel'][idx],limits[:,0],limits[:,1])
jump=warm[25]-warm[24]
within=np.diff(warm[:25],axis=0)
rank=np.argsort(np.abs(jump))[::-1][:6]
last_plan_reconstructed=np.concatenate([tr['planned_target'][3795:3800],warm[:25]])
first_cross={}
for label,values,threshold in [('original_root_gt_20cm',root_original,.2),('original_yaw_gt_15deg',yaw_original,15),('declared_leg_control_rmse_gt_015',legs,.15),('actual_tilt_gt_025rad',tilt,.25)]:
    found=np.flatnonzero(values[3700:3800]>threshold)
    first_cross[label]=None if not len(found) else dict(control=3700+int(found[0]),post_state_time_s=(3701+int(found[0]))*.02,value=float(values[3700+found[0]]))

result=dict(method='Saved arrays and receipts only; no inference, physics, optimization or hidden alignment.',
    source_hashes={str(p):sha(p) for p in [RUN/'trace.npz',RUN/'plans.json',RUN/'report.json',RUN/'request.json',warm_path,motion_path,BUNDLE/'contract.json',BUNDLE/'pico/original29.npz']},
    trace_counts=dict(controls=len(frame),source_controls=int(np.sum((frame>=361)&(frame<6141))),physics=len(tr['physics_torque'])),
    metric_conventions=dict(original_root='Actual post-control root versus unchanged original29 same source_frame, world XYZ; no alignment.',original_heading='Wrapped yaw difference same source frame.',declared_joint='Saved errors to v4 floor retarget; per-control 12-leg/10-arm RMS, not original29 intent.',feet='Saved declared-reference ankle body-origin world error; not original-relative-feet acceptance metric.',first_crossing='Pointwise diagnostic in reviewed window only; not replacement aggregate acceptance gate.',nominal_cert='Accepted saved optimizer horizon is feasible under nominal hard rollout; each actually committed control separately passed exact full-native ten-step copy. Do not equate nominal H30 with full-native H30 certificate.'),
    windows_10_controls=[segment(i,i+10) for i in range(3700,3800,10)],windows_50_controls=[segment(3700,3750),segment(3750,3800)],
    pointwise_crossings_in_window=first_cross,plan_timeline=recent,failed_request=failed,
    shifted_horizon=dict(requests=21,rejections=len(shifts),all_rejections_only_appended_last_five=all(v['control']>=25 for v in shifts),all_retained_25_prefixes_pass=True,tail_frame_indices=idx.tolist(),tail_matches_direct_feedforward_bit_exact=bool(np.array_equal(warm[25:],feed)),tail_max_difference=float(np.max(np.abs(warm[25:]-feed))),
        boundary_target_delta_rad=jump.tolist(),boundary_target_delta_rms_rad=float(np.sqrt(np.mean(jump**2))),boundary_target_delta_max_abs_rad=float(np.abs(jump).max()),largest_boundary_changes=[dict(joint=names[j],delta_rad=float(jump[j]),retained_target=float(warm[24,j]),new_feedforward_target=float(warm[25,j])) for j in rank],
        retained_target_adjacent_delta_rms_rad=stats(np.sqrt(np.mean(within**2,axis=1))),retained_target_adjacent_delta_max_rad=float(np.abs(within).max()),
        last_3795_full_target_reconstruction='First five executed planned targets plus first25 shifted3800 targets. No far-horizon states/K reconstructed or claimed.'),
    interpretation=['All twenty executed plans retain a feasible nominal incumbent and all100 imminent native-control certificates pass. The failed3800 request executes zero rejected controls.',
        'Loss of shifted-horizon feasibility repeatedly occurs after retained25 controls, but this alone cannot assign cause to padding: viability beyond the retained prefix is not established.',
        'Parent reports exact3800 hold-tail also fails at516ms right ankle pitch. Therefore reject simple padding-only diagnosis and do not recommend another tail sweep.',
        'Next bounded authorized test is sequential restoration: guided optimization final proposal becomes K=0 refinement initial iterate, while keeping original shifted seed regularization anchor and unchanged tracking cost, limits and tolerances. Require ordinary final nominal and full-native certificates before use; result not claimed by this read-only analysis.'],
    limitations=['Tracking distributions use only reviewed final100 controls, not whole-source qualification. Root independently audits original task points/full physical trace.', 'Target discontinuities are commanded-target changes, not actual joint-speed violations.', 'No claim of recursive feasibility or real-time control; no control changes made.'])
hold_path=NEW/'pico_final3800_hold_tail_v1/report.json'
hold=json.loads(hold_path.read_text())['results'][0]
result['separate_exact_actual3800_hold_test']=dict(path=str(hold_path),sha256=sha(hold_path),performed_by='sim agent; read-only report inspection here',hard_nominal_pass=hold['hard_nominal_pass'],hard_nominal_first_failure=hold['hard_nominal_first_failure'],full_native_pass=hold['full_native_oracle']['feasible'],full_native_first_failure=hold['full_native_oracle']['first_failure'])
result['gain_context']=dict(zero_gain_committed_plan_controls=[c for c in range(3700,3800,5) if np.all(tr['feedback_gain'][c:c+5]==0)],last_plan_gain_max_abs=float(np.abs(tr['feedback_gain'][3795:3800]).max()),interpretation='Large final K does not imply a large applied correction; saved raw feedback remains near zero in final second. No causal gain claim.')
np.savez(OUT/'derived_arrays.npz',control=np.arange(3700,3800),original_root_error=root_original[3700:3800],original_yaw_error_deg=yaw_original[3700:3800],declared_leg_rmse=legs[3700:3800],declared_arm_rmse=arms[3700:3800],original_shifted_3800_targets=warm,last_3795_planned_targets_reconstructed=last_plan_reconstructed)
(OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(windows=result['windows_50_controls'],crossings=first_cross,tail={k:result['shifted_horizon'][k] for k in ['rejections','tail_matches_direct_feedforward_bit_exact','boundary_target_delta_rms_rad','boundary_target_delta_max_abs_rad','largest_boundary_changes']},all_final_nominal=all(p['final_accepted_nominal_horizon_feasible'] for p in recent),all_imminent=all(p['all_imminent_native_checks_pass'] for p in recent)),indent=2))
