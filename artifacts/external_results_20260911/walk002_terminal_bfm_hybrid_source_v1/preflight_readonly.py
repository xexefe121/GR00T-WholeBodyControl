"""Read frozen assets and pure-array history only; never construct policy/plant."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import runner

p=Path(__file__).parent
args=SimpleNamespace(**{k:Path(v) if k not in ('clip','switch_control','requested_controls','extension_controls') else v
    for k,v in json.loads((p/'arguments.json').read_text()).items()})
receipt,source,request,endpoint=runner.preflight(args)
sys.path.insert(0,str(args.repo))
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory,state_and_terms
history=BFMHistory();previous=np.zeros(23,np.float32)
contract=json.loads((args.bundle/'contract.json').read_text())
for control in range(1118):
    q,dq=source['qpos'][control],source['qvel'][control]
    state,terms=state_and_terms(q[7:],dq[6:],q[3:7],dq[3:6],previous,np.asarray(contract['default_q']))
    runner.exact(previous,source['fresh_seed_previous_action'][control],'all1118 actual previous actions')
    runner.exact(history.before_update(terms),source['fresh_seed_measured_history'][control],'all1118 measured histories')
    previous=((source['target'][control]-np.asarray(contract['default_q']))*np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))).astype(np.float32)
for key,source_key in (('qpos','qpos'),('qvel','qvel')):runner.exact(endpoint[key],source[source_key][1117],'endpoint control boundary '+key)
runner.exact(endpoint['time'],source['physics_time'][11170],'endpoint switch clock')
runner.exact(endpoint['warning_counts'],source['physics_warning_number'][11170],'endpoint warnings')
runner.exact(endpoint['warning_lastinfo'],source['physics_warning_lastinfo'][11170],'endpoint lastinfo')
assert endpoint['final_integration'].shape==(291,)
result=dict(kind='read_only_frozen_hybrid_preflight',all_pass=True,hash_count=len(receipt['hashes']),
    all1118_prior_actions_and_histories_bitexact=True,root1117_endpoint_q_dq_time_warnings_match=True,
    native_state_size=291,policy_or_physics_constructed=False,inference_optimizer_dynamics_executed=False,
    runner_sha256=hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),
    manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
runner.write(p/'preflight_result.json',result)
print(json.dumps(result))
