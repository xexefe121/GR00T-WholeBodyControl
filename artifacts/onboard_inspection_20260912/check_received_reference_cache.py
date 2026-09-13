"""Compare cached packet inputs with the full owned-window computation."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet
from gear_sonic.utils.g1_true23_causal_receiver import CausalReceiver
from gear_sonic.utils.g1_true23_received_features import prepare_reference

BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
meta=json.loads((BASE/'causal_dynamics_v1/bank/bank.json').read_text())
results=[]
for clip in ('walk002','walk003','pico','walk008'):
    model,c,motion,original,timeline=load_case(clip)
    receiver=CausalReceiver(c,model=model,tasks=meta['tasks'],standing_qpos=timeline['configured_standing_qpos'],now=0.)
    peak=0.;checks=0
    def compare(now):
        global peak,checks
        actual=receiver.reference(now)
        owned={k:np.stack([s[k] for s in receiver.samples]) for k in receiver.samples[0]}
        tasks=dict(source_task_position_w=np.stack([t[0] for t in receiver.tasks]),
                   source_task_quaternion_wxyz=np.stack([t[1] for t in receiver.tasks]))
        expected=prepare_reference(owned,tasks,c)
        for key in expected:
            np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=1e-12)
            peak=max(peak,float(np.abs(actual[key]-expected[key]).max()))
        checks+=1
    for sequence in range(len(motion['joint_pos'])):
        fields={k:v[sequence] for k,v in motion.items() if k!='fps'}
        assert receiver.receive(Packet(0,sequence,sequence*.02,fields),original['source_task_position_w'][sequence],
            original['source_task_quaternion_wxyz'][sequence],sequence*.02)
        if sequence<45 or sequence%37==0:compare(sequence*.02)
    # Exercise generated braking references after a real timeout. No fresh
    # packets are admitted until explicit rearm.
    now=sequence*.02+.2
    for step in range(110):compare(now+step*.02)
    assert receiver.gate.fault and receiver.mode=='latched_standing'
    results.append(dict(clip=clip,comparisons=checks,maximum_error=peak,generated_stopping_checked=True))
    print(json.dumps(results[-1]),flush=True)
output=BASE/'onboard_factory_firmware_v1/received_reference_cache_check.json'
output.write_text(json.dumps(dict(cases=results,passed=True,future_samples_used=False),indent=2))
