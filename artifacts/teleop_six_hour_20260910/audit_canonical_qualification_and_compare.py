"""Independent raw-trace checks and matched-setting BFM goal comparison."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
BASE=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
OUT=BASE/'canonical_qualification_independent_review_v1'
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA,MODEL,PHYSICS,load_motion


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def write(path,value):
    assert not path.exists()
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


spec=importlib.util.spec_from_file_location('frozen_intent_inspector',OUT/'inspect_bfm_tracking.py')
inspector=importlib.util.module_from_spec(spec);spec.loader.exec_module(inspector)
native_path=ROOT.parent/'GR00T-WholeBodyControl'/MODEL
source_model_path=ROOT.parent/'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'
native=mujoco.MjModel.from_xml_path(str(native_path));source=mujoco.MjModel.from_xml_path(str(source_model_path))
contract=json.loads((ART/'mjbatch_native23_inputs_v1/contract.json').read_text())
limits=np.asarray(contract['joint_limits']);np.testing.assert_array_equal(limits,native.jnt_range[1:])
kp,kd,effort,velocity=(np.asarray(contract[k]) for k in ('kp','kd','native_effort','native_velocity'))

# Extract completion expressions from the actual frozen auditor, not a mirror implementation.
tree=ast.parse((OUT/'qualify_recorded_candidate.py').read_text())
completion={node.targets[0].id:node.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
    and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('full_source','full_lifecycle')}
witness=[]
for name,count,steps,source_stop,total,expect in (
    ('last_source_slot_partial',10,[10]*9+[7],10,10,(False,False)),
    ('source_complete_return_partial',12,[10]*11+[7],10,12,(True,False)),
    ('entire_lifecycle_complete',12,[10]*12,10,12,(True,True)),
    ('source_short',9,[10]*9,10,12,(False,False))):
    scope=dict(count=count,np=np,substeps=np.asarray(steps),phase={'control_stop':source_stop},motion={'joint_pos':[0]*(total+11)})
    actual=tuple(bool(eval(compile(ast.Expression(completion[k]),'<frozen-auditor-expression>','eval'),scope)) for k in ('full_source','full_lifecycle'))
    assert actual==expect
    witness.append(dict(name=name,source_pass=actual[0],lifecycle_pass=actual[1],expected=list(expect),passed=True))

rows=[];raw_reviews=[]
for clip in ('pico','walk002'):
    original,timeline,original_path=load_motion(clip)
    phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
    orig29_path=DATA/('pico_freedancing_v1/optical_reference_v2/original29.npz' if clip=='pico' else 'walk002/original_source_bundle_v1/original_reference.npz')
    orig29=load(orig29_path)
    for variant,directory in (
        ('original',ART/f'bfm_{clip}_feedback_v2'),
        ('A',BASE/'bfm_original_canonical_v1'/clip),
        ('B',BASE/'bfm_original_canonical_floor_v1'/clip),
        ('v4',BASE/'bfm_floor_v4_v1'/clip)):
        report=json.loads((directory/'report.json').read_text())
        assert report['position_gain']==1 and report['yaw_gain']==2 and report['goal_horizon']==8 and report['arm_reference'] is False
        assert report.get('leg_error_gain',0)==0 and report.get('residual_checkpoint') is None
        assert report.get('arm_intent_ik',False) is False and report.get('ankle_barrier') is None
        assert report.get('goal_gyro_convention','published-world-unscaled')=='published-world-unscaled'
        archive=load(directory/'trace.npz');count=len(archive['qpos'])-1
        a,b=phase['control_start'],min(count,phase['control_stop'])
        current_audit_path=directory/'recorded_source_audit_v1.json'
        if current_audit_path.exists():
            audit=json.loads(current_audit_path.read_text());intent=audit['original_intent']
            assert audit['trace_sha256']==sha(directory/'trace.npz') and audit['report_sha256']==sha(directory/'report.json')
        else:
            assert variant=='original'
            audit=None;intent=inspector.inspect(directory,output_name=None)
        assert intent['controls']==b-a and intent['trace_sha256']==sha(directory/'trace.npz')
        if variant in ('A','B'):
            # Independently supplement two missing checks in the frozen root auditor.
            steps=len(archive['physics_torque'])
            assert archive['physics_torque'].shape==(steps,23) and steps==count*10
            assert archive['physics_qpos'].shape==(steps+1,30) and archive['physics_qvel'].shape==(steps+1,29)
            assert np.max(np.abs(np.linalg.norm(archive['physics_qpos'][:,3:7],axis=1)-1))<1e-10
            assert report['engine_warning_counts']==[0]*8
            predicted=np.clip(kp*(np.repeat(archive['target'],10,axis=0)-archive['physics_qpos'][:-1,7:])-kd*archive['physics_qvel'][:-1,6:],-effort,effort)
            torque_error=float(np.max(np.abs(predicted-archive['physics_torque'])))
            assert torque_error<1e-10
            range_error=np.maximum(0,np.maximum(limits[:,0]-archive['physics_qpos'][:,7:],archive['physics_qpos'][:,7:]-limits[:,1]))
            range_max=float(range_error.max());speed_max=float(np.max(np.abs(archive['physics_qvel'][:,6:])/velocity))
            effort_max=float(np.max(np.abs(archive['physics_torque'])/effort))
            assert range_max==audit['actual_range_excess_rad'] and speed_max==audit['actual_velocity_ratio'] and effort_max==audit['actual_effort_ratio']
            np.testing.assert_array_equal(archive['qpos'],archive['physics_qpos'][::10])
            np.testing.assert_array_equal(archive['qvel'],archive['physics_qvel'][::10])
            assert abs(report['simulated_seconds']-steps*.002)<1e-8
            pose=archive['qpos'][a+1:b+1]
            original_root=orig29['source_qpos29'][a+11:b+11,:3]
            root95=float(np.percentile(np.linalg.norm(pose[:,:3]-original_root,axis=-1),95))
            assert abs(root95-intent['original_root_world_p95_m'])<1e-12
            source_feet=[];native_original_feet=[]
            data=mujoco.MjData(native);source_data=mujoco.MjData(source)
            for control in range(a,b):
                frame=control+11;data.qpos[:]=archive['qpos'][control+1];source_data.qpos[:]=orig29['source_qpos29'][frame]
                mujoco.mj_kinematics(native,data);mujoco.mj_kinematics(source,source_data)
                actual=np.array([data.xpos[native.body(s+'_ankle_roll_link').id]-data.qpos[:3] for s in ('left','right')])
                wanted=np.array([source_data.xpos[source.body(s+'_ankle_roll_link').id]-source_data.qpos[:3] for s in ('left','right')])
                native_wanted=original['body_pos_w'][frame,[6,12]]-original['body_pos_w'][frame,0]
                source_feet.append(actual-wanted);native_original_feet.append(actual-native_wanted)
            original_native_feet95=np.percentile(np.linalg.norm(native_original_feet,axis=-1),95,axis=0)
            np.testing.assert_allclose(original_native_feet95,intent['world_axis_relative_foot_p95_m'],atol=1e-12,rtol=0)
            source_feet95=np.percentile(np.linalg.norm(source_feet,axis=-1),95,axis=0)
            raw_reviews.append(dict(clip=clip,variant=variant,physics_steps=steps,torque_shape=list(archive['physics_torque'].shape),
                raw_torque_reconstruction_max_error=torque_error,zero_engine_warnings_verified=True,unit_root_quaternions_verified=True,
                raw_physics_ranges_match_auditor=True,raw_physics_speed_effort_match_auditor=True,clock_and_decimation_exact=True,
                original_root_p95_matches=True,original_native_relative_feet_p95_matches=True,
                original29_root_relative_feet_p95_m=source_feet95.tolist(),
                original_native_root_relative_feet_p95_m=original_native_feet95.tolist(),audit_sha256=sha(current_audit_path)))
        rows.append(dict(clip=clip,variant=variant,directory=str(directory),source_controls=b-a,source_requested=phase['requested_controls'],
            source_complete=b==phase['control_stop'],lifecycle_complete=count==timeline['total_requested_controls'] and report['failure'] is None,
            reported_failure=report['failure'],reference_path=report['reference_path'],reference_sha256=intent['motion_sha256'],
            trace_sha256=sha(directory/'trace.npz'),report_sha256=sha(directory/'report.json'),
            matched_setting_evidence='Position1/yaw2/horizon8, no arm override/residual/ankle barrier; published-world-unscaled goal convention.',
            original_root_world_p95_m=intent['original_root_world_p95_m'],adapted_goal_root_world_p95_m=report['source_metrics']['root_p95'],
            original_root_yaw_p95_deg=intent['original_root_yaw_abs_p95_deg'],
            adapted_native_root_relative_feet_p95_m=intent['world_axis_relative_foot_p95_m'],
            original29_relative_hands_head_p95_m=intent['original_hand_head_relative_p95_m'],
            legs_rmse_against_requested_native_reference_rad=report['source_metrics']['leg_rmse'],
            arms_rmse_against_requested_native_reference_rad=report['source_metrics']['arm_rmse'],
            range_excess_rad=audit['actual_range_excess_rad'] if audit else report['range_excess_max'],
            speed_ratio=audit['actual_velocity_ratio'] if audit else report['velocity_ratio_max'],
            effort_ratio=audit['actual_effort_ratio'] if audit else report['effort_ratio_max'],
            physical_evidence='raw2ms trace independently audited' if variant in ('A','B') else ('root raw2ms recorded-source auditor' if audit else 'producer aggregates only; original trace lacks raw2ms physics and clock ledger'),
            original_baseline_missing_historical_provenance=variant=='original',
            recorded_source_pass=audit['recorded_source_tracking_pass'] if audit else False,
            failed_gates=audit['failed_gates'] if audit else None,
            adapted_reference_labels={'original':'Original native poses/velocity conventions',
                'A':'Exact original native poses, declared central derivative convention',
                'B':'A plus common Z floor lift; same original native joint poses',
                'v4':'v3 multistart joint/root retarget plus common Z floor lift'}[variant]))

# Evidence search is bounded to task BFM report directories, not unrelated archives.
v3_hashes={clip:sha(BASE/'mjbatch_intent_inputs_v1'/clip/'reference.npz') for clip in ('pico','walk002')}
matched_v3=[];examined=[]
for root in (ART,BASE):
    for folder in root.glob('bfm*'):
        if not folder.is_dir():continue
        paths=[folder/'report.json',folder/'pico/report.json',folder/'walk002/report.json']
        for path in paths:
            if not path.exists():continue
            record=json.loads(path.read_text());ref=record.get('motion_override') or record.get('reference_path')
            clip=record.get('clip')
            if clip not in v3_hashes or not ref or not Path(ref).exists():continue
            reference_hash=sha(ref);examined.append(dict(report=str(path),reference_sha256=reference_hash))
            if reference_hash==v3_hashes[clip] and record.get('arm_reference') is False and record.get('position_gain')==1 and record.get('yaw_gain')==2 and record.get('goal_horizon')==8:
                matched_v3.append(str(path))
assert not matched_v3,'New matched v3 physical evidence found; include it rather than claiming absence.'
result=dict(kind='independent_canonical_qualification_and_matched_goal_comparison',producer_sha256=sha(__file__),
    frozen_auditor_sha256=sha(OUT/'qualify_recorded_candidate.py'),frozen_inspector_sha256=sha(OUT/'inspect_bfm_tracking.py'),
    acceptance_sha256=sha(OUT/'SIM_ACCEPTANCE.md'),partial_completion_witnesses=witness,
    raw_A_B_checks=raw_reviews,comparison=rows,matched_v3_physical_runs=matched_v3,matched_v3_evidence_search=examined,
    reviewer_findings=[
        'Frozen root auditor does not enforce physics_torque shape (N,23); independent supplement verifies all four real A/B archives have correct shape.',
        'Frozen root auditor records engine_warning_counts but does not gate nonzero counts; all four real A/B archives independently verified zero.',
        'Feet gate is against the requested native reference in world axes relative to root. It is original-native invariant for A/B commonZ, but v3/v4 change the native relative foot task and need explicit comparison labels.',
        'A partial final source slot cannot pass full-source or full-lifecycle, but source_controls counts that slot; treat it as slots unless all10 substeps are known.',
        'Original baseline aggregate physical extrema are not equivalent to a newly audited raw2ms trace.'],
    no_dynamics_rerun=True,hardware_authorized=False,full_body_teleoperation_qualified=False)
write(OUT/'comparison.json',result)
(OUT/'comparison_producer_snapshot.py').write_bytes(Path(__file__).read_bytes())
lines=['All four original-pose canonical A/B candidates fail recorded-source qualification. A improves original root position relative to the original baseline, but none passes root/heading/feet/legs/hands together. B adds no qualification success and the PICO run ends before the full source finishes.','',
    '| Clip | Goal | Source controls | Original root p95 (m) | Original yaw p95 (deg) | Native feet p95 L/R (m) | Original hands p95 L/R (m) | Original head p95 (m) | Leg RMSE (rad) | Range excess (rad) |',
    '|---|---|---:|---:|---:|---|---|---:|---:|---:|']
for r in rows:
    feet=r['adapted_native_root_relative_feet_p95_m'];hands=r['original29_relative_hands_head_p95_m']
    source=f"{r['source_controls']}/{r['source_requested']}"
    lines.append(f"|{r['clip']}|{r['variant']}|{source}|{r['original_root_world_p95_m']:.3f}|{r['original_root_yaw_p95_deg']:.2f}|{feet[0]:.3f}/{feet[1]:.3f}|{hands[0]:.3f}/{hands[1]:.3f}|{hands[2]:.3f}|{r['legs_rmse_against_requested_native_reference_rad']:.3f}|{r['range_excess_rad']:.6f}|")
lines+=['',
    'Original = original native poses and original valid velocity conventions. A = the exact original poses with explicitly central derivatives. B = A plus the declared causal floor-lift pose filter. v4 = the different v3 multistart joint/root retarget plus floor lift. Full matched no-arm v3 physical evidence was not found; v3 exists only in this matched comparison as counterfactual inference, so no closed-loop row is invented.', '',
    'The root/yaw/hands/head columns always use original29 intent, without alignment. Native foot and leg columns use each declared native reference. A/B preserve original-native relative feet and joint positions exactly; v3/v4 change these tasks. Adapted-goal root errors remain available separately in JSON. PICO B metrics cover only4763/5780 source controls and cannot be ranked as full-source performance.', '',
    'All listed policies have evidenced position1/yaw2/horizon8, no arm override, residual, or ankle barrier. Original baseline action parity was previously reproduced exactly in the counterfactual diagnostic. The older baseline folders lack historical provenance/raw2ms physics and simulated-time ledgers; their range/speed/effort values are producer aggregates, while A/B have independent raw-step checks and v4 has the root raw-step auditor. All original report/trace/reference bytes remain unchanged.', '',
    'Independent A/B checks confirm every raw torque shape is(N,23), all root quaternions are unit, all warning counts are zero, control snapshots exactly match every tenth physics sample, clocks match2ms steps, PD torque reconstructs exactly, raw bound/speed/effort extrema match the root auditor, and original-root/native-relative-foot metrics agree. Four AST-derived completion witnesses reject incomplete final source or lifecycle slots.', '',
    'Review defects were sent to root without editing shared scripts: the frozen auditor lacks an explicit torque shape check and does not gate engine warnings. Neither defect changes these four results because the independent checks pass. Source-control counts may include a partial terminal slot even though completion gates reject it; the actual four BFM traces have complete10-step slots. The snapshot hash identifies the exact reviewed version.', '',
    'No received-stream, real-time, sensor-only, or hardware qualification follows from these recorded runs. No dynamics were rerun for this audit.', '']
(OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),rows=len(rows),raw_reviews=len(raw_reviews),all_A_B_recorded_source_pass=False),indent=2))
