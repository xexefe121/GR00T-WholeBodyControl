"""One-frame rescue probes retain the failed rollout's previous-pose step box."""
import json
from pathlib import Path
import sys
import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT,DATA,MODEL,PHYSICS,load_motion
from gear_sonic.utils.g1_true23_intent_retarget import Native23LegIK,torso_matched_base

BASE=ROOT/'artifacts/teleop_six_hour_20260910'
with np.load(BASE/'intent_retarget_v3/pico/kinematic_diagnostics.npz',allow_pickle=False) as z:
    prior={key:z[key].copy() for key in z.files}
with np.load(DATA/'pico_freedancing_v1/optical_reference_v2/original29.npz',allow_pickle=False) as z:
    original=z['source_qpos29'].copy()
motion,_,_=load_motion('pico')
native=mujoco.MjModel.from_xml_path(str(ROOT.parent/'GR00T-WholeBodyControl'/MODEL))
source=mujoco.MjModel.from_xml_path(str(ROOT.parent/'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'))
nd,sd=mujoco.MjData(native),mujoco.MjData(source)
feet=[source.body(side+'_ankle_roll_link').id for side in ('left','right')]
torso=source.body('torso_link').id
solver=Native23LegIK(native,posture_weight=.002)
vlim=np.asarray(json.loads((ROOT/PHYSICS).read_text())['physics']['velocity_limit_hardware_radps'])
rows=[]
for frame in (659,664,747,1202,3737):
    sd.qpos[:]=original[frame]
    mujoco.mj_kinematics(source,sd)
    seed=np.r_[motion['body_pos_w'][frame,0],motion['body_quat_w'][frame,0],motion['joint_pos'][frame]]
    base=torso_matched_base(native,nd,seed,sd.xpos[torso],sd.xmat[torso].reshape(3,3))
    previous=prior['qpos'][frame-1]
    result,error,_,_=solver.solve_with_height(base,sd.xpos[feet].copy(),sd.xmat[feet].reshape(2,3,3).copy(),
                                              posture_qpos=previous,previous_qpos=previous,max_step_rad=.8*vlim[:12]*.02)
    rows.append(dict(frame=frame,old_foot_error=prior['foot_position_error'][frame].tolist(),new_foot_error=error.tolist(),
                     old_height_shift_m=float(prior['qpos'][frame,2]-base[2]),new_height_shift_m=float(result[2]-base[2]),
                     adjacent_leg_ratio_max=float(np.max(np.abs(result[7:19]-previous[7:19])/.02/vlim[:12])),
                     solver=solver.last_solution_info))
    assert rows[-1]['adjacent_leg_ratio_max']<=.8+1e-9
(BASE/'intent_retarget_multistart_recovery_probe.json').write_text(json.dumps(rows,indent=2))
print(json.dumps([{key:row[key] for key in ('frame','old_foot_error','new_foot_error','old_height_shift_m','new_height_shift_m','adjacent_leg_ratio_max')}
                  |dict(selected=row['solver']['selected_start'],costs=[a['cost'] for a in row['solver']['attempts']]) for row in rows],indent=2))
