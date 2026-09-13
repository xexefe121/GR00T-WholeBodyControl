"""Complete shared reference checks and explicitly transfer the tested actor."""
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive
from gear_sonic.utils.g1_true23_direct_body_goal import ReceivedBodyGoal
from gear_sonic.utils.g1_true23_received_features import features_numpy

BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1')


def main():
    bank=BASE/'focused_walk002_task_closure_bank_v1'
    packed=archive(bank/'bfm_reference_inputs_v1.npz')
    assert bool(packed['task_closure_stand'])
    reports=[]
    for clip_id,clip in enumerate(('walk003','walk002','pico','walk008')):
        _,contract,motion,original,_=load_case(clip);ref=archive(bank/(clip+'.npz'))
        receiver=SimpleNamespace(gate=SimpleNamespace(epoch=0,fault=None),samples=deque(maxlen=39),tasks=deque(maxlen=39))
        goal=ReceivedBodyGoal(contract,task_closure_stand=True);errors=[];alphas=[]
        for frame in range(len(ref['joint'])):
            receiver.samples.append({k:v[frame] for k,v in motion.items() if k!='fps'})
            receiver.tasks.append((original['source_task_position_w'][frame],original['source_task_quaternion_wxyz'][frame]))
            q,v=ref['states'][frame,:30],ref['states'][frame,30:]
            base=features_numpy(q,v,ref,frame,contract['default_q'],np.zeros(23),np.zeros(300))
            actual=goal.features(base,receiver)
            np.testing.assert_array_equal(actual[:1323],base)
            errors.append(float(np.max(np.abs(actual[1323:1748]-packed[f'body_{clip_id}'][frame]))))
            alphas.append(actual[-1])
        np.testing.assert_allclose(alphas,packed[f'alpha_{clip_id}'],rtol=0,atol=1e-6)
        assert max(errors)<2e-4,(clip,max(errors))
        reports.append(dict(clip=clip,frames=len(alphas),body_max_error=max(errors),blend_max_error=float(np.max(np.abs(np.array(alphas)-packed[f'alpha_{clip_id}'])))))
    output=BASE/'motion_curriculum_initial_v1';output.mkdir(exist_ok=False)
    old=BASE/'focused_walk002_pilot_v2/actor_00040.pt'
    saved=torch.load(old,map_location='cpu',weights_only=False)
    request={**saved['request'],'task_closure_stand':True,'initialization_source':str(old),
        'actor_weights_unchanged':True,'external_reference_semantics_changed':True}
    torch.save(dict(actor=saved['actor'],request=request,initialization_only=True),output/'actor_initial.pt')
    (output/'preparation.json').write_text(json.dumps(dict(cases=reports,original_actor=str(old),actor_weights_unchanged=True,
        physics_performed=False,simulation_qualified=False),indent=2)+'\n')
    print(json.dumps(dict(output=str(output),cases=reports)),flush=True)


if __name__=='__main__':main()
