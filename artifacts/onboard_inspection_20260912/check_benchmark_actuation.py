"""Check torque conversion and actual command history on recorded physical states."""
from pathlib import Path
from types import SimpleNamespace
import json,sys
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.utils.g1_true23_benchmark_actuation import NativeBenchmarkGains
from gear_sonic.utils.g1_true23_received_features import MeasuredHistory
from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand

fw=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
c=json.loads((ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text(encoding='utf-8'))
for key in ('default_q','kp','kd','training_effort','joint_limits'):c[key]=np.asarray(c[key])
cfg=yaml.safe_load((fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml').read_text(encoding='utf-8'))
ids=list(range(13))+list(range(15,20))+list(range(22,27))

class RecordedPolicy:
    def __init__(self):
        self.kp=np.asarray(cfg['joint_kp'])[ids];self.kd=np.asarray(cfg['joint_kd'])[ids]
        self.policy=SimpleNamespace(default=np.asarray(cfg['default_joint_q'])[ids],previous=np.zeros(12))
        self.receiver=SimpleNamespace(history=MeasuredHistory(c));self.target=None
    def command(self,q,v,now):return ControllerCommand(self.target.copy(),{})
    def commit_applied(self,q,v,target):self.receiver.history.commit(q,v,target)

base=RecordedPolicy();adapter=NativeBenchmarkGains(base,c)
errors=[];inverse=[];clipped=0;count=0
for clip in ('walk002','walk003','pico','walk008'):
    with np.load(fw/'command_space_ppo_v1/eval_bootstrap'/clip/'trace.npz',allow_pickle=False) as z:
        n=min(len(z['qpos']),len(z['qvel']),len(z['target']))
        for i in np.linspace(0,n-1,32,dtype=int):
            q,v,t=z['qpos'][i],z['qvel'][i],z['target'][i]
            raw=adapter.benchmark_equivalent(t,q,v)
            before=base.kp*(t-q[7:])-base.kd*v[6:]
            after=adapter.kp*(raw-q[7:])-adapter.kd*v[6:]
            errors.append(float(np.max(np.abs(before-after))))
            inverse.append(float(np.max(np.abs(adapter.factory_equivalent(raw,q,v)-t))))
            base.target=t
            command=adapter.command(q,v,i*.02)
            clipped+=command.status['native_benchmark_gains']['clipped_targets'];count+=23
            adapter.commit_applied(q,v,command.targets)
            expected=(command.targets-np.asarray(c['default_q']))*np.asarray(c['kp'])/(.25*np.asarray(c['training_effort']))
            np.testing.assert_allclose(base.receiver.history.prior,expected,atol=1e-5,rtol=1e-6)
            equivalent=adapter.factory_equivalent(command.targets,q,v)
            np.testing.assert_allclose(base.policy.previous,equivalent[:12]-base.policy.default[:12],atol=1e-12,rtol=1e-12)
assert max(errors)<1e-10 and max(inverse)<1e-12
out=fw/'benchmark_gains_check_v1';out.mkdir(exist_ok=False)
result=dict(passed=True,physical_state_samples=len(errors),unclipped_torque_max_error=max(errors),
    inverse_target_max_error=max(inverse),clipped_target_components=clipped,total_target_components=count,
    actual_received_history_preserved=True,full_motion_passed=False,hardware_commands=False)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
