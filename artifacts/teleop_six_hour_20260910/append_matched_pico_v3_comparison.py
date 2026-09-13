"""Append newly completed matched PICO v3 evidence without changing prior audits."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
OLD=BASE/'canonical_qualification_independent_review_v1'
OUT=BASE/'canonical_qualification_comparison_v2'
OUT.mkdir(exist_ok=False)
CASE=BASE/'bfm_intent_v3_matched_v1/pico'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


old=json.loads((OLD/'comparison.json').read_text())
audit=json.loads((CASE/'recorded_source_audit_v2.json').read_text())
report=json.loads((CASE/'report.json').read_text());intent=audit['original_intent']
assert audit['audit_revision']==2 and audit['trace_sha256']==sha(CASE/'trace.npz') and audit['report_sha256']==sha(CASE/'report.json')
assert report['position_gain']==1 and report['yaw_gain']==2 and report['goal_horizon']==8 and report['arm_reference'] is False
assert not report['arm_intent_ik'] and report['residual_checkpoint'] is None and report['ankle_barrier'] is None
assert report['goal_gyro_convention']=='published-world-unscaled'
v3ref=Path(report['motion_override']);v4case=BASE/'bfm_floor_v4_v1/pico'
v4report=json.loads((v4case/'report.json').read_text())
v4ref=Path(v4report['motion_override'])
assert sha(v3ref)==sha(v4ref.parent/'before_floor_reference.npz')
assert sha(CASE/'runner_snapshot.py')==sha(v4case/'runner_snapshot.py')
for path in (CASE,v4case):
    r=json.loads((path/'report.json').read_text());assert r['engine_warning_counts']==[0]*8
    with np.load(path/'trace.npz') as z: assert z['physics_torque'].shape==(r['completed']*10,23)
row=dict(clip='pico',variant='v3',directory=str(CASE),source_controls=audit['source_controls'],source_requested=audit['source_requested_controls'],
    source_complete=audit['gates']['full_original_source'],lifecycle_complete=audit['gates']['full_lifecycle'],reported_failure=report['failure'],
    reference_path=str(v3ref),reference_sha256=sha(v3ref),trace_sha256=sha(CASE/'trace.npz'),report_sha256=sha(CASE/'report.json'),
    matched_setting_evidence='Position1/yaw2/horizon8, no arm override/residual/barrier; same runner snapshot hash as v4; v4 before-floor reference byte-identical v3.',
    original_root_world_p95_m=intent['original_root_world_p95_m'],adapted_goal_root_world_p95_m=report['source_metrics']['root_p95'],
    original_root_yaw_p95_deg=intent['original_root_yaw_abs_p95_deg'],adapted_native_root_relative_feet_p95_m=intent['world_axis_relative_foot_p95_m'],
    original29_relative_hands_head_p95_m=intent['original_hand_head_relative_p95_m'],
    legs_rmse_against_requested_native_reference_rad=report['source_metrics']['leg_rmse'],arms_rmse_against_requested_native_reference_rad=report['source_metrics']['arm_rmse'],
    range_excess_rad=audit['actual_range_excess_rad'],speed_ratio=audit['actual_velocity_ratio'],effort_ratio=audit['actual_effort_ratio'],
    physical_evidence='Root revision2 raw2ms qualification auditor',recorded_source_pass=audit['recorded_source_tracking_pass'],
    failed_gates=audit['failed_gates'],adapted_reference_labels='v3 multistart original29-intent joint/root retarget; canonical derivatives, no floor lift')
rows=old['comparison']+[row]
order={'original':0,'A':1,'B':2,'v3':3,'v4':4}
rows.sort(key=lambda r:(r['clip']!='pico',order[r['variant']]))
v4=next(r for r in rows if r['clip']=='pico' and r['variant']=='v4')
deltas={key:float(v4[key]-row[key]) for key in ('original_root_world_p95_m','original_root_yaw_p95_deg',
    'legs_rmse_against_requested_native_reference_rad','arms_rmse_against_requested_native_reference_rad')}
deltas['original_hands_head_p95_m']=(np.asarray(v4['original29_relative_hands_head_p95_m'])-row['original29_relative_hands_head_p95_m']).tolist()
result=dict(kind='matched_original_canonical_retarget_BFM_comparison_revision2',comparison=rows,
    previous_comparison_path=str(OLD/'comparison.json'),previous_comparison_sha256=sha(OLD/'comparison.json'),
    added_v3_audit_path=str(CASE/'recorded_source_audit_v2.json'),added_v3_audit_sha256=sha(CASE/'recorded_source_audit_v2.json'),
    v3_v4_same_runner_sha256=sha(CASE/'runner_snapshot.py'),v4_before_floor_matches_v3=True,
    v3_v4_source_controls_equal_5780=True,v3_v4_raw_torque_shape_and_zero_warning_checks_passed=True,
    v4_minus_v3_matched_pico_metric_changes=deltas,walk002_matched_v3_physical_evidence_available=False,
    auditor_revision2_tightening='Requires exact(N,23) torque dimensions and eight integer zero engine-warning counts; old v1 audit results preserved. Actual A/B and v3/v4 pass these evidence checks but fail tracking gates.',
    previous_files_changed=False,no_dynamics_rerun=True,producer_sha256=sha(__file__),recorded_source_pass_any=False,
    full_body_teleoperation_qualified=False,hardware_authorized=False)
(OUT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
(OUT/'producer_snapshot.py').write_bytes(Path(__file__).read_bytes())
lines=['No compared BFM candidate passes recorded-source full-body tracking acceptance. The newly matched PICO v3 run completes the lifecycle and respects physical limits, but original heading/hand/foot/leg tracking remains poor. Its floor-lift counterpart v4 also fails.', '',
    '| Clip | Goal | Source controls | Original root p95 (m) | Original yaw p95 (deg) | Native feet p95 L/R (m) | Original hands p95 L/R (m) | Original head p95 (m) | Leg RMSE (rad) | Range excess (rad) |',
    '|---|---|---:|---:|---:|---|---|---:|---:|---:|']
for r in rows:
    f=r['adapted_native_root_relative_feet_p95_m'];h=r['original29_relative_hands_head_p95_m']
    lines.append(f"|{r['clip']}|{r['variant']}|{r['source_controls']}/{r['source_requested']}|{r['original_root_world_p95_m']:.3f}|{r['original_root_yaw_p95_deg']:.2f}|{f[0]:.3f}/{f[1]:.3f}|{h[0]:.3f}/{h[1]:.3f}|{h[2]:.3f}|{r['legs_rmse_against_requested_native_reference_rad']:.3f}|{r['range_excess_rad']:.6f}|")
lines+=['',
    'Original uses original native poses and valid original velocity conventions. A preserves all original poses and explicitly changes derivatives to central intervals. B adds common Z floor lift to A. v3 uses the separate multistart original29-intent joint/root retarget. v4 adds floor lift to that v3 reference. Root/yaw/hands/head columns always score original29 intent. Feet/legs use the declared native reference; A/B preserve those original-native relative tasks, whereas v3/v4 change them. No time or heading alignment is applied.', '',
    'The matched PICO v3/v4 pair has exactly the same6530 lifecycle controls/5780 source controls, position1/yaw2/horizon8, frozen runner hash and absence of arm/residual/barrier overrides. The byte-preserved v4 before-floor input matches the v3 reference. Floor lift changes original root p95 by+0.005m, yaw p95 by−7.53deg, and left/right hand p95 by−0.052/−0.002m; neither crosses acceptance. Leg RMSE remains about0.195rad, feet about0.28–0.33m, and original hand errors about0.49–0.58m. No matched walk002 v3 physical run is asserted.', '',
    'PICO B covers only4763/5780 source controls and is explicitly incomplete. Its metrics are not a full-source success comparison. Older original baselines have producer physical aggregates without raw2ms/clock ledgers or historical provenance sidecars. A/B have separate independent raw2ms torque/clock/quaternion/limit checks; v3 has the revision2 auditor, and v4 has the earlier raw-step auditor plus independently verified torque shape and zero warnings. Existing v1 evidence remains unchanged.', '',
    'Auditor revision2 now rejects malformed torque dimensions and missing/nonzero engine-warning evidence. Ten independent AST-extracted negative witnesses confirm those checks. The four A/B outcomes remain failures after tightening; their real traces already satisfy these evidence requirements.', '',
    'The matched v3 run was unpaced:183.7s wall for130.6s simulation and inference p95 34.66ms on a loaded host. That is not a20ms timing pass. No new dynamics were run while assembling this comparison. None of these results qualifies received-stream teleoperation, sensor-only control, or hardware.', '']
(OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),rows=len(rows),v4_minus_v3=deltas),indent=2))
