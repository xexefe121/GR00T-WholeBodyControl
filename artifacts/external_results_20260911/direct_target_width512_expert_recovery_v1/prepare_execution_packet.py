"""Concrete read-only input freeze/launcher preparation; never dispatches recovery."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def subject(p):return dict(path=str(p).replace('\\','/'),sha256=sha(p))
def wsl_path(p):
    value=Path(p).as_posix().replace('\\','/')
    if len(value)<3 or value[1:3]!=':/' or not value[0].isalpha():
        raise ValueError('Expected an absolute Windows drive path')
    return '/mnt/'+value[0].lower()+value[2:]
def write(p,obj):
    with p.open('x',newline='\n') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def main():
    prep=read(BASE/'source_preparation_v2.json');review_path=NEW/'direct_target_width512_expert_recovery_review_v1/source_review.json'
    review=read(review_path);assert sha(review_path)=='771a5ec79b84039d805533cd09446f1eb8c32370ec0b148e0c52919af05be28f' and review['source_review_pass'] is True
    selection=read(BASE/'inputs/selection_receipt.json');assert selection['passed'] is True
    assert read(BASE/'input_preparation_exit.json')['exit_code']==0
    for name,digest in selection['outputs'].items():assert sha(BASE/'inputs'/name)==digest
    sources=prep['source_sha256']
    for name,digest in sources.items():assert sha(BASE/'source_snapshot_v1'/name)==digest
    proposal=read(BASE/'execution_request.proposal.json')
    request=dict(proposal,root_selected=True,source_only=False,actual_input_preparation_run=True,actual_recovery_run=False,
        dispatch_requires_separate_exact_root_concrete_clearance=True)
    request.pop('required_future_subjects')
    roles=request['subjects']
    for name,p in {
        'source_preparation':BASE/'source_preparation_v2.json','source_review':review_path,
        'boundary_review':NEW/'direct_target_width512_expert_recovery_review_v1/boundary_review.json',
        'selected_snapshot':BASE/'inputs/precontrol251.npz','selected_prefix':BASE/'inputs/actual_prefix251.npz',
        'input_selection':BASE/'inputs/selection_receipt.json','input_preparation_request':BASE/'input_preparation_request.json',
        'input_preparation_exit':BASE/'input_preparation_exit.json',
    }.items():roles[name]=subject(p)
    request['environment']['PYTHONPATH']=wsl_path(BASE/'source_snapshot_v1')
    write(BASE/'execution_request.json',request)
    pins=read(BASE/'inherited_input_catalog.json')['input_sha256']
    for role in roles.values():pins[role['path']]=role['sha256']
    shell=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh')
    pins[str(shell)]=sha(shell)
    for p,digest in list(pins.items()):
        actual=sha(p)
        if actual!=digest:raise ValueError('Input changed during concrete freeze: '+p)
    frozen=dict(kind='one_fixed_actual_width81000_pre251_expert_recovery',source_sha256=sources,input_sha256=pins,
        protocol=request['protocol'],source_preparation=subject(BASE/'source_preparation_v2.json'),source_review=subject(review_path),
        request_sha256=sha(BASE/'execution_request.json'))
    write(BASE/'frozen_inputs.json',frozen)
    launchpins=dict(pins)
    for name,digest in sources.items():launchpins[str(BASE/'source_snapshot_v1'/name)]=digest
    for name in ('execution_request.json','frozen_inputs.json','run_recovery_durable.ps1','run_recovery.sh','verify_recovery_completed.py','prepare_execution_packet.py','launcher_parse.json','bash_parse_v2.json','test_execution_helpers.py','execution_helper_tests_final_v2.xml'):
        p=BASE/name;launchpins[str(p)]=sha(p)
    lb=wsl_path(BASE)
    args=['-d','Ubuntu-22.04','--cd','/','--','bash',wsl_path(shell),'env',
          'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
          'PYTHONPATH='+lb+'/source_snapshot_v1','bash',lb+'/run_recovery.sh']
    write(BASE/'launch_receipt.json',dict(kind='one_fixed_actual_pre251_expert_recovery',request_sha256=sha(BASE/'execution_request.json'),
        frozen_receipt_sha256=sha(BASE/'frozen_inputs.json'),launcher_sha256=sha(BASE/'run_recovery_durable.ps1'),
        wsl_arguments=args,input_sha256=launchpins,source_review_sha256=sha(review_path),
        dispatch_authorized=False,root_concrete_clearance_required=True))
    result=dict(request=subject(BASE/'execution_request.json'),frozen=subject(BASE/'frozen_inputs.json'),
        launch_receipt=subject(BASE/'launch_receipt.json'),launcher=subject(BASE/'run_recovery_durable.ps1'),
        selection=subject(BASE/'inputs/selection_receipt.json'),snapshot=subject(BASE/'inputs/precontrol251.npz'),
        prefix=subject(BASE/'inputs/actual_prefix251.npz'),frozen_inputs=len(pins),sources=len(sources),launch_pins=len(launchpins),
        actual_recovery_dispatched=False)
    write(BASE/'concrete_preparation.json',result);print(json.dumps(result))
if __name__=='__main__':main()
