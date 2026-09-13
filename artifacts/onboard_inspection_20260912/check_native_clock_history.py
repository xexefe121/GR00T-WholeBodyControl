"""Compare plant-imported native memory against the verified direct recurrence."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import mujoco
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.utils.g1_true23_factory_conditioned import NativeFactoryConditionedController
from gear_sonic.utils.g1_true23_received_features import MeasuredHistory

fw=Path(r'E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1')
c=json.loads((ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text())
for k in ('default_q','kp','kd','training_effort'):c[k]=np.asarray(c[k])
cfg=yaml.safe_load((fw/'decoded_configs/policies/mimic_test/fsm_mimic_test.yaml').read_text())
default=np.asarray(cfg['default_joint_q'],np.float32)
memory=MeasuredHistory(c)
controller=SimpleNamespace(receiver=SimpleNamespace(history=memory,contract=c),default=default,
    cfg=cfg,history=np.zeros((5,76),np.float32),previous=np.zeros(23,np.float32),initialized=False)
z=np.load(fw/'native23_trainable_v1/zero_conditioning_lifecycle/trace.npz')
bank=np.load(fw.parent/'causal_dynamics_v1/bank/walk002.npz')
q=np.r_[bank['states'][10,:30][None],z['qpos'][:-1]]
v=np.r_[bank['states'][10,30:][None],z['qvel'][:-1]]
direct=np.zeros((5,76),np.float32);previous=np.zeros(23,np.float32);peak=0.
for i in range(100):
    rot=np.empty(9);mujoco.mju_quat2Mat(rot,q[i,3:7])
    dq=v[i,6:].copy();dq[[4,5,10,11]]=0
    obs=np.r_[v[i,3:6]*.25,-rot.reshape(3,3)[2],q[i,7:]-default,dq*.05,previous,0].astype(np.float32)
    if i==0:direct[:]=obs
    else:direct[-1]=obs
    NativeFactoryConditionedController.import_native_history(controller,i)
    if i==0:controller.history[:]=obs
    else:controller.history[-1]=obs
    peak=max(peak,float(np.abs(controller.history-direct).max()),float(np.abs(controller.previous-previous).max()))
    np.testing.assert_allclose(controller.history,direct,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(controller.previous,previous,atol=2e-6,rtol=2e-6)
    target=z['target'][i]
    memory.commit(q[i],v[i],target)
    previous=(target-default).astype(np.float32)
    direct[:-1]=direct[1:]
result={'controls':100,'maximum_native_memory_error':peak,'passed':peak<2e-6,
    'including_initial_padding':True,'actual_applied_history':True}
(fw/'factory_clock_v1/history_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

from gear_sonic.utils.g1_true23_factory_locomotion import FactoryHumanLoco12,FactoryLocomotionTeleop
actor=fw/'human_loco_trainable_v1/factory_loco12.onnx'
left=FactoryHumanLoco12(fw,c['joint_limits'],actor)
right=FactoryHumanLoco12(fw,c['joint_limits'],actor)
memory=MeasuredHistory(c)
adapter=SimpleNamespace(receiver=SimpleNamespace(history=memory,contract=c),policy=right)
z=np.load(fw/'human_loco_received_v3/walk002/trace.npz')
q=np.r_[bank['states'][10,:30][None],z['qpos'][:-1]]
v=np.r_[bank['states'][10,30:][None],z['qvel'][:-1]]
peak=target_peak=0.
for i in range(900):
    FactoryLocomotionTeleop.import_native_history(adapter,i)
    rot=np.empty(9);mujoco.mju_quat2Mat(rot,q[i,3:7]);gravity=-rot.reshape(3,3)[2]
    command=np.array([.2 if 250<=i<500 else 0,0,.4 if i>=600 else 0])
    a=left.step(q[i],v[i],gravity,command);b=right.step(q[i],v[i],gravity,command)
    peak=max(peak,float(np.abs(left.history-right.history).max()))
    target_peak=max(target_peak,float(np.abs(a-b).max()))
    np.testing.assert_allclose(left.history,right.history,atol=2e-6,rtol=2e-6)
    # Include a held previous command to test importing an actual late action.
    target=z['target'][i-1 if i%17==0 and i else i]
    memory.commit(q[i],v[i],target)
    left.previous[:]=target[:12]-left.default[:12]
result={'controls':900,'maximum_native_memory_error':peak,'maximum_target_error':target_peak,
    'includes_held_previous_commands':True,'passed':peak<2e-6 and target_peak<2e-5}
assert result['passed'],result
(fw/'factory_clock_v1/locomotion_history_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
