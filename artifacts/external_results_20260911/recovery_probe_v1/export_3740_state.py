"""Export actual integration state and verify restored physical continuation."""
from pathlib import Path
import sys
import json
import copy
BASE=Path(__file__).parent
sys.path.insert(0,str(BASE/"source_snapshot"))
import mujoco
import numpy as np
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states,sha256
ROOT=Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")
TASK=Path("/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910")
BUNDLE=ROOT/"artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
PRODUCER=TASK/"mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1"
native,c,original,timeline,manifest=load_native_bundle(BUNDLE,"pico")
motion,_=load_motion_override(TASK/"mjbatch_intent_floor_inputs_v1/pico/reference.npz",BUNDLE,"pico",native,c,original,timeline,manifest)
with np.load(PRODUCER/"trace.npz") as a:source={k:a[k].copy() for k in ("target","physics_qpos","physics_qvel","physics_torque")}
with np.load(BASE/"3740_source_goals_yaw2.npz") as a:expected={k:a[k].copy() for k in ("target","physics_qpos","physics_qvel","physics_torque")}
kp,kd,effort=[np.asarray(c[k]) for k in ("kp","kd","native_effort")]
d=mujoco.MjData(native);initial=motion_states(motion)[10];d.qpos[:]=initial[:30];d.qvel[:]=initial[30:];mujoco.mj_forward(native,d)
for control,target in enumerate(source["target"][:3740]):
    for sub in range(10):
        d.ctrl[:]=np.clip(kp*(target-d.qpos[7:])-kd*d.qvel[6:],-effort,effort)
        mujoco.mj_step(native,d);step=control*10+sub+1
        np.testing.assert_array_equal(d.qpos,source["physics_qpos"][step]);np.testing.assert_array_equal(d.qvel,source["physics_qvel"][step])
        np.testing.assert_array_equal(d.ctrl,source["physics_torque"][step-1])
    mujoco.mj_kinematics(native,d)
spec=mujoco.mjtState.mjSTATE_INTEGRATION
vector=np.empty(mujoco.mj_stateSize(native,spec));mujoco.mj_getState(native,d,vector,spec)
np.savez_compressed(BASE/"actual_3740_integration_state.npz",state_spec=int(spec),state_vector=vector,
    qpos=d.qpos.copy(),qvel=d.qvel.copy(),qacc_warmstart=d.qacc_warmstart.copy(),time=d.time,ctrl=d.ctrl.copy(),
    targets=expected["target"],source_control=3740,source_frame=3751)
restored=mujoco.MjData(native);mujoco.mj_setState(native,restored,vector,spec);mujoco.mj_forward(native,restored)
np.testing.assert_array_equal(restored.qacc_warmstart,d.qacc_warmstart)
np.testing.assert_array_equal(restored.qpos,d.qpos);np.testing.assert_array_equal(restored.qvel,d.qvel)
for control,target in enumerate(expected["target"]):
    for sub in range(10):
        restored.ctrl[:]=np.clip(kp*(target-restored.qpos[7:])-kd*restored.qvel[6:],-effort,effort)
        mujoco.mj_step(native,restored);step=control*10+sub+1
        np.testing.assert_array_equal(restored.qpos,expected["physics_qpos"][step])
        np.testing.assert_array_equal(restored.qvel,expected["physics_qvel"][step])
        np.testing.assert_array_equal(restored.ctrl,expected["physics_torque"][step-1])
    mujoco.mj_kinematics(native,restored)
report=dict(mujoco=mujoco.__version__,actual_prefix_physics_steps=37400,prefix_bitexact=True,
    exported_spec="mjSTATE_INTEGRATION",state_spec=int(spec),state_size=len(vector),
    restore="MjData(native); mj_setState(native,data,state_vector,mjtState(state_spec)); mj_forward(native,data) once; then native manualPD without state writes",
    warmstart_unchanged_by_initial_forward=True,restored_continuation_physics_steps=300,
    restored_qpos_qvel_torque_bitexact_to_full_MjData_branch=True,
    native_model_bundle=str(BUNDLE),native_arrays_sha256=sha256(BUNDLE/"prepared_model_arrays.npz"),
    snapshot_sha256=sha256(BASE/"actual_3740_integration_state.npz"),probe_sha256=sha256(BASE/"3740_source_goals_yaw2.npz"),script_sha256=sha256(__file__))
with (BASE/"actual_3740_state_restoration.json").open("x") as f:json.dump(report,f,indent=2);f.write("\n")
print(json.dumps(report),flush=True)
