"""Read-only cross-check of the independent walk003 WSL replay and final second."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
CASE = E / 'bfm_online_intent_v2/walk003_allmargin_wsl_independent_replay_v1'
PRODUCER = E / 'mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1'
REFERENCE = E / 'mjbatch_intent_floor_inputs_v1/walk003/reference.npz'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with np.load(path, allow_pickle=False) as a: return {k:a[k].copy() for k in a.files}
def stats(a): return dict(p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),maximum=float(np.max(a)))


report = json.loads((CASE/'report.json').read_text())
audit = json.loads((CASE/'recorded_source_audit_v2.json').read_text())
assert sha(CASE/'report.json') == audit['report_sha256']
assert sha(CASE/'trace.npz') == audit['trace_sha256'] == report['trace_sha256']
assert sha(REFERENCE) == audit['reference_sha256'] == report['reference_sha256']
assert audit['recorded_source_tracking_pass'] and all(audit['gates'].values())
for name,digest in [('qualify_recorded_candidate.py',audit['audit_sha256']),
                    ('inspect_bfm_tracking.py',audit['original_intent']['inspector_sha256']),
                    ('SIM_ACCEPTANCE.md',audit['acceptance_sha256'])]:
    source = Path(__file__).with_name(name)
    assert sha(source) == digest
    (CASE/('qualification_'+name)).write_bytes(source.read_bytes())
trace,producer,motion = read(CASE/'trace.npz'),read(PRODUCER/'trace.npz'),read(REFERENCE)
comparisons = {}
for name in ('qpos','qvel','physics_qpos','physics_qvel','physics_torque','target','source_frame','physics_substeps'):
    np.testing.assert_array_equal(trace[name],producer[name])
    comparisons[name] = True
np.testing.assert_array_equal(trace['physics_warning_counts'],np.zeros((15691,8),dtype=np.int64))
np.testing.assert_array_equal(trace['physics_warning_lastinfo'],np.zeros((15691,8),dtype=np.int64))
np.testing.assert_array_equal(trace['qpos'],trace['physics_qpos'][::10])
np.testing.assert_array_equal(trace['qvel'],trace['physics_qvel'][::10])
np.testing.assert_array_equal(trace['source_frame'],np.arange(1569)+11)
assert np.max(np.abs(trace['physics_time']-np.arange(15691)*.002)) < 1e-8
frames = trace['source_frame']
desired_root = np.repeat(motion['body_pos_w'][frames,0],10,axis=0)[-500:]
desired_joints = np.repeat(motion['joint_pos'][frames],10,axis=0)[-500:]
actual = trace['physics_qpos'][-500:]
velocity = trace['physics_qvel'][-500:]
rooterror = np.linalg.norm(actual[:,:3]-desired_root,axis=1)
joint_error = actual[:,7:]-desired_joints
final = dict(interval='last 500 post-step states, (30.380,31.380] seconds',physics_samples=500,
    reference_convention='contemporaneous declared 50Hz reference held for each ten 2ms steps',
    reference_root_variation_max_m=float(np.max(np.abs(desired_root-desired_root[0]))),
    reference_joint_variation_max_rad=float(np.max(np.abs(desired_joints-desired_joints[0]))),
    root_error_m=stats(rooterror),root_error_final_m=float(rooterror[-1]),
    maximum_absolute_joint_speed_radps=stats(np.max(np.abs(velocity[:,6:]),axis=1)),
    joint_speed_rms_radps=float(np.sqrt(np.mean(velocity[:,6:]**2))),
    final_maximum_absolute_joint_speed_radps=float(np.max(np.abs(velocity[-1,6:]))),
    final_maximum_joint_speed_name=json.loads((ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text())['joint_names'][int(np.argmax(np.abs(velocity[-1,6:])))],
    joint_tracking_rmse_rad=float(np.sqrt(np.mean(joint_error**2))),
    root_linear_speed_mps=stats(np.linalg.norm(velocity[:,:3],axis=1)),
    root_angular_speed_radps=stats(np.linalg.norm(velocity[:,3:6],axis=1)),
    root_height_min_m=float(actual[:,2].min()),root_height_max_m=float(actual[:,2].max()),
    final_joint_velocity_radps=velocity[-1,6:].tolist(),
    lifecycle_completed=True,stationary_zero_velocity_claim=False,
    limitation='All lifecycle frames completed, but nonzero balancing/joint motion remains; no added settle threshold or extra simulation time.')
result = dict(kind='independent_saved_array_verification_and_final_second_review',
    input_hashes={str(p):sha(p) for p in (CASE/'report.json',CASE/'trace.npz',CASE/'recorded_source_audit_v2.json',
        PRODUCER/'trace.npz',REFERENCE,Path(__file__))}, all_comparison_fields_bit_exact=comparisons,
    every_2ms_warning_counts_and_lastinfo_exact_zero=True,final_second=final,
    distinction='This new pinned WSL MuJoCo3.2.3 actual-target replay reproduces full nominal producer physics exactly. It does not recompute saved K. Root Windows saved-K feedback replay separately failed at control570; that feedback robustness result is not overturned.',
    recorded_source_audit_pass=True,online_controller_pass=False,hardware_authorized=False)
(CASE/'saved_evidence_and_final_second_review_v1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(CASE/'saved_evidence_reviewer_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result,indent=2,allow_nan=False))
