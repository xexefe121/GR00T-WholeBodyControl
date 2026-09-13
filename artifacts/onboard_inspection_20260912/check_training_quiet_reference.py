"""Verify full-body motion does not receive a quiet-standing training label."""
from pathlib import Path
import sys,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import received_reference_stationary

torch.set_num_threads(1)
shapes=dict(root_velocity=(3,),joint_velocity=(23,),root_omega=(3,),
    feet_velocity=(2,3),task_velocity=(3,3),task_omega=(3,3))
r={k:torch.zeros((7,*shape)) for k,shape in shapes.items()}
for row,key in enumerate(shapes,1):r[key][row].flatten()[0]=.2
observed=received_reference_stationary(r).tolist()
assert observed==[True,False,False,False,False,False,False],observed

fw=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
bank=fw.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1'
rows=[]
for item in json.loads((bank/'bank.json').read_text())['clips']:
    with np.load(bank/item['file'],allow_pickle=False) as z:
        ref={k:torch.from_numpy(z[k].astype(np.float32)) for k in shapes}
        old=(ref['root_velocity'].norm(dim=-1)<.02)&(ref['joint_velocity'].abs().amax(-1)<.05)
        new=received_reference_stationary(ref)
        assert not (new&~old).any()
        changed=(old&~new).nonzero().flatten()
        source=(changed>=item['source_start']+11)&(changed<item['source_stop']+11)
        rows.append(dict(clip=item['name'],changed_controls=changed.sub(11).tolist(),
            changed_source_controls=int(source.sum()),previous_quiet=int(old.sum()),corrected_quiet=int(new.sum())))
out=fw/'quiet_reference_check_v1';out.mkdir(exist_ok=False)
result=dict(passed=True,isolated_motion_cases=6,stationary_case_preserved=True,recordings=rows,
    explains_existing_motion_failures=False,checkpoint_behavior_changed=False,training_started=False)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
