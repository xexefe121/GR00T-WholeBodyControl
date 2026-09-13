"""Exact restored continuation and fault/rearm checks on private simulation."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
if sys.platform!='win32':
    for p in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps','/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):sys.path.append(p)
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from gear_sonic.utils.g1_true23_native_targets import NativeTargetController
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--library',type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    base=Path('E:/codex-artifacts' if sys.platform=='win32' else '/mnt/e/codex-artifacts')
    fw=base/'sonic23_teleop_resume_20260911/onboard_factory_firmware_v1'
    bank=fw.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1'
    model,c,motion,original,timeline=load_case('walk002');meta=json.loads((bank/'bank.json').read_text())
    controller=NativeTargetController(fw/'native_target_ppo_v2/actor_00185.onnx',fw,c,model=model,tasks=meta['tasks'],
        standing_qpos=timeline['configured_standing_qpos'],now=-.22,native_preview_guard=True,native_standing_capture=True,
        native_preview_delay_substeps=6,native_preview_library=args.library)
    class ObservedSession:
        def __init__(self,session):self.session=session;self.last=None
        def run(self,names,inputs):
            self.last=inputs['features'].copy()
            return self.session.run(names,inputs)
    controller.session=ObservedSession(controller.session)
    d=mujoco.MjData(model)
    with np.load(bank/'walk002.npz') as z:d.qpos[:]=z['states'][10,:30];d.qvel[:]=z['states'][10,30:]
    mujoco.mj_forward(model,d)
    def receive(frame,now):
        return controller.receive(Packet(controller.receiver.gate.epoch,frame,frame*.02,
            {k:v[frame] for k,v in motion.items() if k!='fps'}),original['source_task_position_w'][frame],
            original['source_task_quaternion_wxyz'][frame],now)
    for f in range(11):assert receive(f,(f-11)*.02)
    target_error=0.;state_error=0.;observation_error=0.;tested=[]
    for control in range(470):
        now=control*.02
        if control<425:assert receive(control+11,now)
        if control in (0,5,350,410,440,460):
            snapshot=controller.snapshot(now);plant=copy.copy(d)
            first=controller.command(d.qpos,d.qvel,now).targets.copy()
            first_observation=controller.session.last.copy()
            controller.commit_applied(d.qpos,d.qvel,first)
            for _ in range(10):
                d.ctrl[:]=np.clip(c['kp']*(first-d.qpos[7:])-c['kd']*d.qvel[6:],-c['native_effort'],c['native_effort']);mujoco.mj_step(model,d)
            expected=np.r_[d.qpos,d.qvel];d=copy.copy(plant)
            assert controller.restore_snapshot(snapshot)==now
            actual=controller.command(d.qpos,d.qvel,now).targets.copy()
            observation_error=max(observation_error,float(abs(first_observation-controller.session.last).max()))
            target_error=max(target_error,float(abs(first-actual).max()))
            controller.commit_applied(d.qpos,d.qvel,actual)
            for _ in range(10):
                d.ctrl[:]=np.clip(c['kp']*(actual-d.qpos[7:])-c['kd']*d.qvel[6:],-c['native_effort'],c['native_effort']);mujoco.mj_step(model,d)
            state_error=max(state_error,float(abs(expected-np.r_[d.qpos,d.qvel]).max()))
            assert target_error==0 and state_error==0 and observation_error==0,(control,target_error,state_error,observation_error)
            tested.append(dict(control=control,fault=controller.receiver.gate.fault is not None))
        else:
            target=controller.command(d.qpos,d.qvel,now).targets
            controller.commit_applied(d.qpos,d.qvel,target)
            for _ in range(10):
                d.ctrl[:]=np.clip(c['kp']*(target-d.qpos[7:])-c['kd']*d.qvel[6:],-c['native_effort'],c['native_effort']);mujoco.mj_step(model,d)
    assert controller.receiver.gate.fault is not None
    history=controller.receiver.history.vector().copy();controller.rearm(10.,d.qpos)
    snap=controller.snapshot(10.);controller.velocity[:]=123
    controller.restore_snapshot(snap)
    assert not controller.receiver.gate.fault and controller.receiver.gate.epoch==1
    np.testing.assert_array_equal(controller.receiver.history.vector(),history)
    incompatible=copy.deepcopy(snap);incompatible.wrapper['physics_dt']=.003
    try:controller.restore_snapshot(incompatible)
    except ValueError:pass
    else:raise AssertionError('incompatible wrapper accepted')
    report=dict(passed=True,target_error=target_error,state_error=state_error,observation_error=observation_error,cases=tested,
        fault_and_rearm_preserved=True,wrapper_mismatch_rejected=True,simulation_ready=False)
    # One restored boundary permits exactly one observation advancement.
    controller.restore_snapshot(snap)
    controller.last_observation_time=10.
    try:controller.observation(d.qpos,d.qvel,10.)
    except ValueError:pass
    else:raise AssertionError('Double observation advancement was accepted')
    report['duplicate_timestamp_rejected']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
