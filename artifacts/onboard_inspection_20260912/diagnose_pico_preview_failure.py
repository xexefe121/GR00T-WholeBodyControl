"""Verify actual applied-command replay before saving integration boundaries."""
import argparse,copy,hashlib,json,platform,sys
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for path in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps','/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):
    sys.path.append(path)
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION


def integration_state(model,data):
    state=np.empty(mujoco.mj_stateSize(model,STATE_SPEC),np.float64)
    mujoco.mj_getState(model,data,state,STATE_SPEC)
    return state


def restore_physics(model,state):
    data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,np.asarray(state,np.float64),STATE_SPEC)
    # mj_step recomputes derived fields; no speculative forward before it.
    return data


def set_pd(data,target,kp,kd,effort,limits,limit_brake=False):
    requested=kp*(target-data.qpos[7:])-kd*data.qvel[6:]
    if limit_brake:
        band=np.minimum(.1,.2*(limits[:,1]-limits[:,0]))
        penetration=data.qpos[7:]-np.clip(data.qpos[7:],limits[:,0]+band,limits[:,1]-band)
        requested-=100*penetration+2*np.where(penetration*data.qvel[6:]>0,data.qvel[6:],0)
    data.ctrl[:]=np.clip(requested,-effort,effort)
    return requested


def first_difference(expected,actual,step,component):
    different=np.flatnonzero(expected!=actual)
    if not len(different):return None
    i=int(different[0])
    return dict(physics_step=step,component=component,index=i,expected=float(expected[i]),actual=float(actual[i]))


def verify_run(run,controls,output):
    report=json.loads((run/'report.json').read_text())
    model,contract,*_=load_case(report['clip'])
    with np.load(run/'trace.npz',allow_pickle=False) as z:
        states=z['states'];targets=z['targets'];torques=z['torques']
    if any(control<0 or control*10>=len(targets) for control in controls):
        raise ValueError('Requested boundary is not available before a recorded step')
    checked=[]
    manifest_path=run/'run_manifest.json'
    if manifest_path.exists():
        for item in json.loads(manifest_path.read_text())['files']:
            path=Path(item['path'])
            if path.suffix in ('.so','.onnx','.xml','.npz') or path.name=='contract.json':
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if digest!=item['sha256']:raise ValueError(f'Pinned physical input changed: {path}')
                checked.append(dict(path=str(path),sha256=digest))
    data=mujoco.MjData(model);data.qpos[:]=states[0,:30];data.qvel[:]=states[0,30:59]
    mujoco.mj_forward(model,data)
    kp=np.asarray(report['actual_pd_kp']);kd=np.asarray(report['actual_pd_kd']);effort=contract['native_effort']
    error=np.zeros((len(targets),3));saved={};first=None
    for step,target in enumerate(targets):
        if step%10==0 and step//10 in controls:saved[step//10]=copy.copy(data)
        set_pd(data,target,kp,kd,effort,contract['joint_limits'],report.get('native_limit_brake',False))
        error[step,2]=np.max(np.abs(data.ctrl-torques[step]))
        if first is None:first=first_difference(torques[step],data.ctrl,step,'commanded_torque')
        mujoco.mj_step(model,data)
        error[step,:2]=[np.max(np.abs(data.qpos-states[step+1,:30])),np.max(np.abs(data.qvel-states[step+1,30:59]))]
        if first is None:first=first_difference(states[step+1,:30],data.qpos,step,'qpos')
        if first is None:first=first_difference(states[step+1,30:59],data.qvel,step,'qvel')
    exact=bool(np.all(error==0));roundtrips=[]
    np.save(output/'every_2ms_replay_error.npy',error)
    if exact:
        for control,original in saved.items():
            state=integration_state(model,original);restored=restore_physics(model,state)
            a=copy.copy(original);b=restored
            for step in range(control*10,min(control*10+16,len(targets))):
                for d in (a,b):
                    set_pd(d,targets[step],kp,kd,effort,contract['joint_limits'],report.get('native_limit_brake',False))
                    mujoco.mj_step(model,d)
                if not np.array_equal(integration_state(model,a),integration_state(model,b)):
                    raise AssertionError(f'Integration serialization diverged at step {step}')
            roundtrips.append(dict(control=control,exact=True,steps=min(16,len(targets)-control*10)))
            np.savez(output/f'boundary_{control}.npz',integration_state=state,state_spec=int(STATE_SPEC),
                qpos=original.qpos,qvel=original.qvel,qacc_warmstart=original.qacc_warmstart,
                previous_target=targets[control*10-1] if control else np.zeros(23),
                previous_target_valid=bool(control),control=control)
    result=dict(input_run=str(run),physics_steps_checked=len(targets),
        initial_state='archived FP64 initial q/v; fresh MjData; mj_forward',
        every_2ms_max_errors=dict(zip(('qpos','qvel','commanded_torque'),error.max(axis=0).tolist())),
        first_difference=first,exact_applied_target_replay=exact,
        snapshots_saved_only_after_exact_replay=True,snapshot_boundaries=sorted(saved) if exact else [],
        serialization_continuation_checks=roundtrips,state_spec='mjSTATE_INTEGRATION',
        mujoco_version=mujoco.__version__,architecture=platform.machine(),python_version=platform.python_version(),
        pinned_files_checked=checked,controller_history_reconstructed=False,
        state_reference='https://mujoco.readthedocs.io/en/3.2.3/programming/simulation.html#state-and-control')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pinned_files_checked'}),flush=True)
    if not exact:raise RuntimeError('Applied-target replay diverged; no exact-continuation snapshots released')
    return model,contract,saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-run',type=Path,required=True)
    p.add_argument('--snapshot-controls',nargs='+',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    verify_run(args.input_run,args.snapshot_controls,args.output)


if __name__=='__main__':main()
