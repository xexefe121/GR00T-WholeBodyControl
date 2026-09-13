"""Read-only final producer inspection; no inference, optimizer or physics."""
import hashlib
import json
from pathlib import Path
import numpy as np

PROCESS=Path(__file__).parent
NEW=PROCESS.parent
OUT=NEW/'walk002_terminal_bfm_hybrid_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
result={}
for name in ('exit_status.json','running.json','pid.json'):
    p=PROCESS/name
    if p.exists():result[name]=json.loads(p.read_text(encoding='utf-8-sig'))
for name in ('report.json','comparison.json','fatal.json','progress.json','verified_switch_receipt.json'):
    p=OUT/name
    if p.exists():
        d=json.loads(p.read_text());result[name]=d if name!='comparison.json' else {k:v for k,v in d.items() if k not in ('lifecycle','extension')}
if (OUT/'comparison.json').exists() and (PROCESS/'exit_status.json').exists():
    receipt=json.loads((PROCESS/'launch_receipt.json').read_text())
    post={p:sha(Path(p))==h for p,h in receipt['verified_hashes'].items()}
    result['all134_launch_files_unchanged']=all(post.values())
    (PROCESS/'postrun_all_launch_hashes.json').write_text(json.dumps(dict(count=len(post),all_exact=all(post.values()),checks=post),indent=2)+'\n')
    main=read(OUT/'trace.npz');source=read(NEW/'walk002_full_control_lm_v1/trace.npz')
    prefix=min(1117,int(np.sum(main['physics_substeps']==10)))
    checks={}
    for key in ('qpos','qvel','target','source_frame','physics_qpos','physics_qvel','physics_requested_torque',
                'physics_torque','physics_actuator_force','physics_time','physics_expected_time','physics_warning_number','physics_warning_lastinfo'):
        if key.startswith('physics_'):
            n=prefix*10+int(key not in ('physics_requested_torque','physics_torque','physics_actuator_force'))
        else:n=prefix+int(key in ('qpos','qvel'))
        checks[key]=exact(main[key][:n],source[key][:n])
    checks['history']=exact(main['history'][:prefix],source['fresh_seed_measured_history'][:prefix])
    checks['previous_action']=exact(main['previous_action'][:prefix],source['fresh_seed_previous_action'][:prefix])
    result['saved_prefix_comparison']=dict(controls=prefix,all_exact=all(checks.values()),checks=checks)
    for key,path in (('main_trace',OUT/'trace.npz'),('main_report',OUT/'report.json'),('main_request',OUT/'request.json'),
                     ('hold_trace',OUT/'post_lifecycle_hold_5s/trace.npz'),('hold_report',OUT/'post_lifecycle_hold_5s/report.json')):
        if path.exists():result[key+'_sha256']=sha(path)
    if (OUT/'post_lifecycle_hold_5s/trace.npz').exists():
        hold=read(OUT/'post_lifecycle_hold_5s/trace.npz')
        continuity={key:exact(main[key],hold['initial_'+key[6:]]) for key in main if key.startswith('final_') and 'initial_'+key[6:] in hold}
        result['main_hold_saved_continuity']=dict(fields=len(continuity),all_exact=all(continuity.values()),checks=continuity)
        result['hold_report']=json.loads((OUT/'post_lifecycle_hold_5s/report.json').read_text())
    (PROCESS/'producer_observation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
