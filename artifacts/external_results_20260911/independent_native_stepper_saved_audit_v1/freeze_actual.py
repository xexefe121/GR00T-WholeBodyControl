"""Bind completed owners and root-cleared source before one saved-only audit."""
import json
import hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
RUN=NEW/'independent_native_stepper_equivalence_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
review_path=NEW/'independent_native_stepper_saved_audit_root_review_v1/review_v2.json'
assert sha(review_path)=='5304a6f655ac967b84eff943d909d05eeb839139b0afbc2462ff31f2aa247a9a'
review=read(review_path);assert review['source_review_passed'] is True and review['actual_audit_selected_once_after_completed_witness_and_replay'] is True
for group in ('source_pins','owner_schema_pins'):
    for path,digest in review[group].items():assert sha(path)==digest,path
config={};subjects={}
for stage,report_sha,owner_sha in [
    ('witness','e24d3e31e738d4a9f0908f1fa76eb8a66b737001d180e0c16b6b1d7fabf8919b','7a81971663909d5cc703de62275189f93ea24844283d420a84a9d1797aebc41c'),
    ('replay','f72a19eb15f240c4a270a984c8d416816c51a2063f7a3a7928e6a0133ec81909','64a89e068eb59f8e2d07200104c42ab887b86f7bcef2dd64790f95c199cf996a')]:
    folder=RUN/(stage+'_process');owner_path=RUN/(stage+'_completion_verification.json');report_path=RUN/stage/'report.json'
    assert sha(owner_path)==owner_sha and sha(report_path)==report_sha
    owner=read(owner_path);report=read(report_path)
    assert owner['passed'] is True and report['passed'] is True and report['error'] is None
    paths=dict(request=RUN/(stage+'_request.json'),clearance=folder/'launch_clearance.json',report=report_path,owner=owner_path,
        process_exit=folder/'exit.json',launch_receipt=folder/'launch_receipt.json',process_absence=folder/'process_absence.json',
        pre_hashes=folder/'prerun_hashes.json',post_hashes=folder/'postrun_hashes.json')
    config[stage]={k:p.as_posix() for k,p in paths.items()}
    config[stage]['completion_evidence']=[(folder/name).as_posix() for name in ['start.json','child.json','raw_exit.json','diagnostic_verdict.json']]
    for name,path in paths.items():subjects[stage+'_'+name]={'path':path.as_posix(),'sha256':sha(path)}
with (BASE/'actual_stages.json').open('x',encoding='utf-8') as f:json.dump(config,f,indent=2);f.write('\n')
gate=dict(root_source_review={'path':review_path.as_posix(),'sha256':sha(review_path)},source_pins=review['source_pins'],subjects=subjects,
    selected_pure_audit_once=True,actual_stage_config_sha256=sha(BASE/'actual_stages.json'),model_calls=0,native_steps=0,optimizer_updates=0)
with (BASE/'actual_gate.json').open('x',encoding='utf-8') as f:json.dump(gate,f,indent=2);f.write('\n')
print(json.dumps({'ready':True,'gate_sha256':sha(BASE/'actual_gate.json')}))
