"""Coherent native sensor transport and legacy ABI check; no timing qualification."""
import argparse,copy,ctypes as ct,json,sys,time
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'artifacts/teleop_resume_20260911'))
from run_causal_native_clock import Bridge,ptr,DOUBLE,INT64,initial_sensor_sample
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    fw=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
    model,c,*_=load_case('walk002');data=mujoco.MjData(model)
    with np.load(fw.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1/walk002.npz') as z:
        data.qpos[:]=z['states'][10,:30];data.qvel[:]=z['states'][10,30:]
    mujoco.mj_forward(model,data);replay=copy.copy(data)
    bridge=Bridge(fw/'factory_clock_v5/libtrue23clock.so')
    try:
        initial=np.zeros(462);initial[:59]=np.r_[data.qpos,data.qvel]
        initial[382:439]=initial_sensor_sample(model,data);initial[439:]=data.qpos[7:]
        bridge.publish_observation(bridge.address,0,0,time.monotonic_ns(),ptr(initial))
        # Old ABI must not overwrite a caller's382-double allocation.
        guarded=np.full(390,-1234567.);ident,fault,stamp=ct.c_int(),ct.c_int(),ct.c_int64()
        assert bridge.lib.clock_read_observation(bridge.address,ct.byref(ident),ct.byref(fault),ct.byref(stamp),ptr(guarded))
        np.testing.assert_array_equal(guarded[382:],-1234567.)
        arrays=[np.ascontiguousarray(c[k],np.float64) for k in ('kp','kd','native_effort','native_velocity','default_q','training_effort')]
        target=data.qpos[7:].copy();states=np.empty((41,60));targets=np.empty((40,23));torques=np.empty((40,23))
        timing=np.empty((40,7),np.int64);summary=np.empty(10);epoch=time.monotonic_ns()+20_000_000
        count=bridge.lib.clock_run(bridge.address,model._address,data._address,4,epoch,*[ptr(x) for x in arrays],
            ptr(target),ptr(states),ptr(targets),ptr(torques),timing.ctypes.data_as(INT64),ptr(summary))
        observation=bridge.observation();control,_,_,values=observation
        assert count==40 and control==3,(count,control)
        before=control*10-1
        np.testing.assert_array_equal(values[383:406],states[before,7:30])
        np.testing.assert_array_equal(values[406:429],states[before,36:59])
        assert values[382]==states[before,59]
        for step in range(before+1):replay.ctrl[:]=torques[step];mujoco.mj_step(model,replay)
        expected=initial_sensor_sample(model,replay)
        # Helper reads post-integration q/time, whereas IMU values are from
        # pre-integration Euler sensor evaluation. Compare IMU blocks only.
        error=float(abs(values[429:439]-expected[47:57]).max())
        assert error<1e-12,error
        np.testing.assert_array_equal(values[439:462],targets[control*10-1])
        report=dict(passed=True,sensor_imu_error=error,sensor_age_s=control*.02-values[382],
            legacy382_buffer_preserved=True,exact_applied_target_preserved=True,
            expected_command_misses=int(summary[1]),timing_qualification=False,simulation_ready=False)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    finally:bridge.close(owner=True)


if __name__=='__main__':main()
