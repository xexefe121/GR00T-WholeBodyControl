"""Restore saved integration state after derived-field forward; verify continuation."""
import json
from pathlib import Path
import sys
BASE=Path(__file__).parent
sys.path.insert(0,str(BASE/"source_snapshot"))
import mujoco
import numpy as np
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,sha256
BUNDLE=Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1")
native,c,*_=load_native_bundle(BUNDLE,"pico")
with np.load(BASE/"actual_3740_integration_state.npz") as a:state={k:a[k].copy() for k in a.files}
with np.load(BASE/"3740_source_goals_yaw2.npz") as a:expected={k:a[k].copy() for k in ("target","physics_qpos","physics_qvel","physics_torque")}
spec=mujoco.mjtState(int(state["state_spec"]))
restored=mujoco.MjData(native)
mujoco.mj_setState(native,restored,state["state_vector"],spec)
mujoco.mj_forward(native,restored)
forward_delta=float(np.max(np.abs(restored.qacc_warmstart-state["qacc_warmstart"])))
# Reapply the saved integration state only during initialization. The forward
# call has built derived geometry/constraint fields but also altered warmstart.
mujoco.mj_setState(native,restored,state["state_vector"],spec)
np.testing.assert_array_equal(restored.qacc_warmstart,state["qacc_warmstart"])
np.testing.assert_array_equal(restored.qpos,state["qpos"]);np.testing.assert_array_equal(restored.qvel,state["qvel"])
kp,kd,effort=[np.asarray(c[k]) for k in ("kp","kd","native_effort")]
for control,target in enumerate(expected["target"]):
    for sub in range(10):
        restored.ctrl[:]=np.clip(kp*(target-restored.qpos[7:])-kd*restored.qvel[6:],-effort,effort)
        mujoco.mj_step(native,restored);step=control*10+sub+1
        np.testing.assert_array_equal(restored.qpos,expected["physics_qpos"][step])
        np.testing.assert_array_equal(restored.qvel,expected["physics_qvel"][step])
        np.testing.assert_array_equal(restored.ctrl,expected["physics_torque"][step-1])
    mujoco.mj_kinematics(native,restored)
report=dict(mujoco=mujoco.__version__,exported_spec="mjSTATE_INTEGRATION",state_spec=int(spec),state_size=len(state["state_vector"]),
    restore="MjData(native); mj_setState(vector,spec); mj_forward(native,data); mj_setState(vector,spec) again before any step; then nativePD with no state writes",
    reason_for_second_initial_state_application="mj_forward changed actual saved qacc_warmstart",naive_forward_warmstart_max_change=forward_delta,
    preserved_failed_v1_script="export_3740_state.py",restored_continuation_physics_steps=300,
    qpos_qvel_torque_bitexact_to_full_actual_MjData_branch=True,
    physical_state_rewrites_after_first_physics_step=0,
    snapshot_sha256=sha256(BASE/"actual_3740_integration_state.npz"),probe_sha256=sha256(BASE/"3740_source_goals_yaw2.npz"),script_sha256=sha256(__file__))
with (BASE/"actual_3740_state_restoration_v2.json").open("x") as f:json.dump(report,f,indent=2);f.write("\n")
print(json.dumps(report),flush=True)
