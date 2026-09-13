"""Compare the recovered pose actor on its training CSV and native23 transfer.

Contract diagnostic only: vendor29 results cannot pass native23 acceptance.
Neither this script nor the controller publishes hardware commands.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import mujoco
import numpy as np
import yaml
from run_factory_mimic_sim import FIRMWARE, ASSETS, REPO, BUNDLE, load_model
sys.path.insert(0, str(REPO))
from gear_sonic.utils.g1_true23_factory_controller import (
    FactoryPoseController, ReceivedPose, ReceivedReference,
    NATIVE23_TO_VENDOR29, quaternion_matrix, normalized_quaternion,
)


def run(args):
    cfg = yaml.safe_load((FIRMWARE/'decoded_configs/policies/cpy_dance/dance.yaml').read_text())
    csv = np.loadtxt(ASSETS/cfg['data_file'].removeprefix('../'), delimiter=',')
    if args.clip != 'factory':
        with np.load(BUNDLE/f'{args.clip}/original29.npz') as z:
            source = z['source_qpos29']
        csv = np.zeros((len(source),71))
        csv[:,:3] = source[:,:3]
        csv[:,3:7] = source[:,[4,5,6,3]]
        csv[:,7:36] = source[:,7:]
        from gear_sonic.utils.g1_true23_factory_controller import quaternion_product
        for i in range(1,len(csv)):
            rotation = quaternion_matrix(source[i,3:7])
            csv[i,36:39] = rotation.T@(source[i,:3]-source[i-1,:3])/.02
            delta = normalized_quaternion(quaternion_product(source[i,3:7],source[i-1,3:7]*np.array([1,-1,-1,-1])))
            sine = np.linalg.norm(delta[1:])
            if sine>1e-9: csv[i,39:42] = rotation.T@(delta[1:]/sine*2*np.arctan2(sine,delta[0]))/.02
            csv[i,42:71] = (source[i,7:]-source[i-1,7:])/.02
    if args.joints == 23:
        m, contract = load_model()
        ids = NATIVE23_TO_VENDOR29
        effort = np.asarray(contract['native_effort'])
    else:
        m = mujoco.MjModel.from_xml_path(str(REPO/'gear_sonic/data/robots/g1/scene_29dof.xml'))
        ids = np.arange(29)
        effort = np.max(np.abs(m.actuator_ctrlrange), axis=1)
    d = mujoco.MjData(m)
    limits = m.jnt_range[1:]
    native_limits = limits if args.joints == 23 else limits[NATIVE23_TO_VENDOR29]
    wrapper = FactoryPoseController(ASSETS/'policies/cpy_dance/Feb13_20-31-05_/actor.onnx',
        FIRMWARE/'decoded_configs/policies/cpy_dance/dance.yaml', native_limits)
    session = wrapper.session
    kp, kd = np.asarray(cfg['kp'])[ids], np.asarray(cfg['kd'])[ids]
    default = np.asarray(cfg['default_dof_pos'])
    d.qpos[:3] = csv[0,:3]
    d.qpos[3:7] = csv[0,[6,3,4,5]]
    d.qpos[7:] = csv[0,7:36][ids]
    mujoco.mj_forward(m,d)
    receiver = ReceivedReference()
    history = np.zeros((5,93), np.float32)
    previous = np.zeros(29, np.float32)
    trace = []
    errors = []
    worst_bound = 0.
    fail = None
    begin = time.perf_counter()
    for frame in range(min(len(csv), int(args.seconds*50))):
        row = csv[frame]
        q29, v29 = np.zeros(29), np.zeros(29)
        q29[ids], v29[ids] = d.qpos[7:], d.qvel[6:]
        rotation = quaternion_matrix(d.qpos[3:7])
        proprio = np.r_[d.qvel[3:6], rotation.T@np.array([0,0,-1]),q29,v29,previous].astype(np.float32)
        if frame == 0: history[:] = proprio
        else:
            history[:-1] = history[1:]
            history[-1] = proprio
        target = row[:71].astype(np.float32).copy()
        if args.joints == 23:
            missing = np.setdiff1d(np.arange(29), ids)
            target[7+missing] = 0
            target[42+missing] = 0
        if args.received:
            target = receiver.accept(ReceivedPose(frame, frame*.02,row[:3],row[[6,3,4,5]],row[7:36][NATIVE23_TO_VENDOR29]))
        quat = normalized_quaternion(d.qpos[3:7])[[1,2,3,0]].astype(np.float32)
        previous = session.run(['actor_actions'], {'proprioception':proprio[None], 'memory':history.reshape(1,-1),
            'quaternion':quat[None], 'target_state':target[None]})[0][0]
        command = (default+previous*.25)[ids]
        if args.joints == 23:
            command = np.clip(command,limits[:,0]+.06,limits[:,1]-.06)
            previous[:] = 0
            previous[ids] = (command-default[ids])/.25
        for sub in range(10):
            torque = kp*(command-d.qpos[7:])-kd*d.qvel[6:]
            if args.joints == 23:
                band = np.minimum(.1,np.diff(limits,axis=1).ravel()*.2)
                penetration = d.qpos[7:]-np.clip(d.qpos[7:],limits[:,0]+band,limits[:,1]-.0-band)
                torque -= 100*penetration + 2*np.where(penetration*d.qvel[6:]>0,d.qvel[6:],0)
            d.ctrl[:] = np.clip(torque,-effort,effort)
            mujoco.mj_step(m,d)
            worst_bound = max(worst_bound,float(np.max(np.maximum(limits[:,0]-d.qpos[7:],d.qpos[7:]-limits[:,1]))))
            tilt = np.arccos(np.clip(quaternion_matrix(d.qpos[3:7])[2,2],-1,1))
            if not np.isfinite(d.qpos).all() or d.qpos[2]<.25 or tilt>1.2:
                fail = {'time':float(d.time),'height':float(d.qpos[2]),'tilt':float(tilt)}
                break
        trace.append(d.qpos.copy())
        errors.append([np.linalg.norm(d.qpos[:2]-row[:2]),np.sqrt(np.mean((d.qpos[7:]-row[7:36][ids])**2))])
        if fail: break
    result = {'joints':args.joints, 'clip':args.clip, 'received':args.received, 'seconds':float(d.time),'requested_seconds':args.seconds,
        'failure':fail,'p95_root_xy_m_joint_rmse_rad':np.quantile(errors,.95,axis=0).tolist(),
        'max_joint_bound_excess_rad':worst_bound,'wall_seconds':time.perf_counter()-begin,'acceptance_run':False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(args.output/'trace.npz',qpos=np.asarray(trace),errors=np.asarray(errors))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--joints',type=int,choices=[23,29],required=True)
    ap.add_argument('--received',action='store_true')
    ap.add_argument('--clip',choices=['factory','walk002','walk003','pico','walk008'],default='factory')
    ap.add_argument('--seconds',type=float,default=20)
    ap.add_argument('--output',type=Path,required=True)
    run(ap.parse_args())
