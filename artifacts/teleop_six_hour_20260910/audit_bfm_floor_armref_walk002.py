"""Strict full-physics and original-source intent audit; no physical replay."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA,MODEL,PHYSICS,load_motion

BASE=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
CASE=BASE/'bfm_floor_v4_armref_v1/walk002'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_npz(p):
    with np.load(p) as z: return {k:z[k].copy() for k in z.files}


report=json.loads((CASE/'report.json').read_text())
intent=json.loads((CASE/'original_intent_metrics.json').read_text())
provenance=json.loads((CASE/'provenance.json').read_text())
trace=read_npz(CASE/'trace.npz')
_,timeline,_=load_motion('walk002')
checks={p:sha(p)==h for p,h in provenance.items()}
assert all(checks.values())
assert sha(CASE.parent/'runner_snapshot.py')==provenance[str(ROOT/'gear_sonic/scripts/evaluate_g1_true23_bfmzero.py')]
native_path=ROOT.parent/'GR00T-WholeBodyControl'/MODEL
source_path=ROOT.parent/'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'
native=mujoco.MjModel.from_xml_path(str(native_path)); source=mujoco.MjModel.from_xml_path(str(source_path))
contract=json.loads((ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text())
limits=np.asarray(contract['joint_limits']); np.testing.assert_array_equal(limits,native.jnt_range[1:])
kp,kd,effort,velocity=(np.asarray(contract[k]) for k in ('kp','kd','native_effort','native_velocity'))
n=len(trace['target']); assert n==report['completed']==1133
assert len(trace['physics_torque'])==n*10
np.testing.assert_array_equal(trace['qpos'],trace['physics_qpos'][::10])
np.testing.assert_array_equal(trace['qvel'],trace['physics_qvel'][::10])
expected=np.clip(kp*(np.repeat(trace['target'],10,axis=0)-trace['physics_qpos'][:-1,7:])-kd*trace['physics_qvel'][:-1,6:],-effort,effort)
np.testing.assert_allclose(expected,trace['physics_torque'],atol=1e-10,rtol=0)
byjoint=np.maximum(limits[:,0]-trace['physics_qpos'][:,7:],trace['physics_qpos'][:,7:]-limits[:,1])
byrange=np.maximum(0,byjoint.max(axis=1))
byspeed=np.max(np.abs(trace['physics_qvel'][:,6:])/velocity,axis=1)
byeffort=np.max(np.abs(trace['physics_torque'])/effort,axis=1)
np.testing.assert_allclose(byrange[1:].reshape(n,10).max(axis=1),trace['range_excess'],atol=1e-12,rtol=0)
np.testing.assert_allclose(byspeed[1:].reshape(n,10).max(axis=1),trace['velocity_ratio'],atol=1e-12,rtol=0)
np.testing.assert_allclose(byeffort.reshape(n,10).max(axis=1),trace['effort_ratio'],atol=1e-12,rtol=0)
worst_step,worst_joint=map(int,np.unravel_index(np.argmax(byjoint),byjoint.shape))
bad=np.flatnonzero(byrange>0)
phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
a,b=phase['control_start'],phase['control_stop']
assert b<=n and b-a==667
original_path=DATA/'walk002/original_source_bundle_v1/original_reference.npz'
original=read_npz(original_path)
nd,sd=mujoco.MjData(native),mujoco.MjData(source)
feet=[]
for control in range(a,b):
    frame=control+11
    nd.qpos[:]=trace['qpos'][control+1]; sd.qpos[:]=original['source_qpos29'][frame]
    mujoco.mj_kinematics(native,nd); mujoco.mj_kinematics(source,sd)
    errors=[]
    for side in ('left','right'):
        name=side+'_ankle_roll_link'
        errors.append((nd.xpos[native.body(name).id]-nd.qpos[:3])-(sd.xpos[source.body(name).id]-sd.qpos[:3]))
    feet.append(errors)
feet=np.asarray(feet)
source_steps=slice(a*10+1,b*10+1)
result=dict(kind='independent_bfm_v4_arm_reference_strict_and_original_intent_audit',clip='walk002',
    controls=n,full_requested_controls=timeline['total_requested_controls'],physics_steps=n*10,
    full_source_controls=667,full_source_completed=True,full_lifecycle_completed=False,failure=report['failure'],
    hashes=dict(trace=sha(CASE/'trace.npz'),report=sha(CASE/'report.json'),original_intent=sha(CASE/'original_intent_metrics.json'),
                native_model=sha(native_path),original29_model=sha(source_path),physics=sha(ROOT/PHYSICS),original29=sha(original_path),
                producer=sha(__file__)),source_hash_checks=checks,
    every_2ms_native_bounds_checked=True,every_2ms_pd_torque_reconstruction_passed=True,
    range_excess_max_rad=float(byrange.max()),range_violation_samples=int(len(bad)),
    first_strict_range_violation_seconds=float(bad[0]*.002),worst_range_violation_seconds=worst_step*.002,
    worst_joint=native.joint(worst_joint+1).name,worst_joint_position=float(trace['physics_qpos'][worst_step,worst_joint+7]),
    velocity_ratio_max=float(byspeed.max()),effort_ratio_max=float(byeffort.max()),
    source_range_excess_max_rad=float(byrange[source_steps].max()),source_velocity_ratio_max=float(byspeed[source_steps].max()),
    source_original29_hand_head_relative_p95_m=intent['original_hand_head_relative_p95_m'],
    source_original29_root_world_p95_m=intent['original_root_world_p95_m'],
    source_original29_yaw_abs_p95_deg=intent['original_root_yaw_abs_p95_deg'],
    source_original29_world_axis_root_relative_foot_p95_m=np.percentile(np.linalg.norm(feet,axis=-1),95,axis=0).tolist(),
    source_leg_joint_rmse=report['source_metrics']['leg_rmse'],source_arm_joint_rmse=report['source_metrics']['arm_rmse'],
    physical_limits_passed=bool(byrange.max()<=1e-6 and byspeed.max()<=1 and byeffort.max()<=1+1e-9),
    full_body_tracking_qualified=False,hardware_authorized=False,
    outcome='Arm joint tracking improves but root/leg/foot errors and lifecycle worsen. Original hand errors remain above15cm. No PICO arm-reference follow-up justified.',
    latency_scope='Shared host unconstrained playback. Source goals use eight poses plus one future pose for canonical velocity support:160ms total pose support. No real-time deadline claim.')
(CASE/'strict_original_intent_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
(CASE/'strict_auditor_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result,indent=2))
