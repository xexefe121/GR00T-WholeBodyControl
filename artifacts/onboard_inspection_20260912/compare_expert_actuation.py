"""Replay one successful native expert under candidate actuator interfaces.

Diagnostic only: saved per-motion targets are never proposed as a live policy.
All branches start from the same full native integration state and then run
without resets or state projection. Original physical limits stay unchanged.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import mujoco
import yaml

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    contract=json.loads((bundle/'contract.json').read_text())
    kp,kd,effort,velocity,limits=[np.asarray(contract[k]) for k in ('kp','kd','native_effort','native_velocity','joint_limits')]
    fw=BASE/'onboard_factory_firmware_v1'
    cfg=yaml.safe_load((fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml').read_text(encoding='utf-8'))
    fkp,fkd=np.array(cfg['joint_kp']),np.array(cfg['joint_kd'])
    if len(fkp)==29:
        ids=list(range(13))+list(range(15,20))+list(range(22,27))
        fkp,fkd=fkp[ids],fkd[ids]
    assert fkp.shape==(23,) and mujoco.__version__=='3.2.3'
    model=mujoco.MjModel.from_xml_path(str(bundle/'native_prepared.xml'))
    with np.load(bundle/'prepared_model_arrays.npz') as z:
        for key in z.files:getattr(model,key)[:]=z[key]
    mujoco.mj_setConst(model,mujoco.MjData(model))
    folder=BASE/'fast_feedback_walk003_v1/baseline_v1'
    traces=[]
    for part in ('nominal','post_lifecycle_hold_5s'):
        with np.load(folder/part/'trace.npz') as z:traces.append({k:z[k].copy() for k in z.files})
    np.testing.assert_array_equal(traces[0]['qpos'][-1],traces[1]['qpos'][0])
    targets=np.concatenate([t['target'] for t in traces])
    expected_q=np.concatenate((traces[0]['physics_qpos'],traces[1]['physics_qpos'][1:]))
    expected_v=np.concatenate((traces[0]['physics_qvel'],traces[1]['physics_qvel'][1:]))
    band=np.minimum(.1,.2*(limits[:,1]-limits[:,0]))
    def brake(q,v):
        penetration=q-np.clip(q,limits[:,0]+band,limits[:,1]-band)
        return 100*penetration+2*np.where(penetration*v>0,v,0)
    reports={}
    for mode in ('native_original','native_margin','native_brake_margin','factory_equivalent_margin'):
        data=mujoco.MjData(model)
        mujoco.mj_setState(model,data,traces[0]['initial_integration'],int(traces[0]['integration_state_spec']))
        original_time=float(data.time);tick=time.monotonic();physical_steps=0;failure=None
        max_q_difference=max_v_difference=max_speed=max_clip=0.;clipped_components=0
        use_brake=mode in ('native_brake_margin','factory_equivalent_margin')
        use_kp,use_kd=(fkp,fkd) if mode=='factory_equivalent_margin' else (kp,kd)
        for control,teacher in enumerate(targets):
            q,v=data.qpos[7:].copy(),data.qvel[6:].copy()
            target=teacher.copy()
            if use_brake:
                target=q+(kp*(teacher-q)+(use_kd-kd)*v+brake(q,v))/use_kp
            if mode!='native_original':
                clipped=np.clip(target,limits[:,0]+.06,limits[:,1]-.06)
                max_clip=max(max_clip,float(abs(clipped-target).max()))
                clipped_components+=int(np.count_nonzero(clipped!=target));target=clipped
            for substep in range(10):
                q,v=data.qpos[7:],data.qvel[6:]
                torque=use_kp*(target-q)-use_kd*v
                if use_brake:torque-=brake(q,v)
                data.ctrl[:]=np.clip(torque,-effort,effort)
                mujoco.mj_step(model,data);physical_steps+=1
                max_q_difference=max(max_q_difference,float(abs(data.qpos-expected_q[physical_steps]).max()))
                max_v_difference=max(max_v_difference,float(abs(data.qvel-expected_v[physical_steps]).max()))
                speed=float((abs(data.qvel[6:])/velocity).max());max_speed=max(max_speed,speed)
                excess=float(np.maximum(limits[:,0]-data.qpos[7:],data.qpos[7:]-limits[:,1]).max())
                tilt=float(np.arccos(np.clip(1-2*np.sum(data.qpos[4:6]**2),-1,1)))
                reasons=[]
                if excess>1e-6:reasons.append('native_joint_bound')
                if speed>1:reasons.append('native_joint_speed')
                if data.qpos[2]<.25 or tilt>1.2:reasons.append('fall')
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():reasons.append('nonfinite')
                if np.any(data.warning.number):reasons.append('engine_warning')
                if reasons:
                    failure=dict(control=control,substep=substep,reasons=reasons,range_excess=excess,speed_ratio=speed,tilt=tilt)
                    break
            if failure:break
        report=dict(mode=mode,simulation_seconds=float(data.time)-original_time,physical_steps=physical_steps,
            complete=failure is None,failure=failure,max_qpos_difference=max_q_difference,max_qvel_difference=max_v_difference,
            max_speed_ratio=max_speed,max_target_clipping=max_clip,clipped_components=clipped_components,
            wall_seconds=time.monotonic()-tick)
        reports[mode]=report;print(json.dumps(report),flush=True)
        (args.output/'report.json').write_text(json.dumps(dict(reports=reports,
            comparison_valid=mode!='native_original' or (report['complete'] and max_q_difference<1e-8 and max_v_difference<1e-7),
            live_controller_tested=False),indent=2)+'\n')
        if mode=='native_original':
            assert report['complete'] and max_q_difference<1e-8 and max_v_difference<1e-7,report
    result=dict(kind='expert_actuation_physical_comparison',clip='walk003',reports=reports,
        saved_teacher_targets=True,live_controller_tested=False,original_physical_limits=True,hardware_commands=False)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
