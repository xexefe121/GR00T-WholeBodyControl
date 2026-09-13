"""Read-only concrete recovery packet verification; never dispatches task work."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
RUN=BASE.parent/'direct_target_width512_expert_recovery_v1'
EXPECTED={
    'execution_request.json':'c3e20cd213091bd9ec0db5a99871404ae609712de5be3e664b6bd844f62edea8',
    'frozen_inputs.json':'954ac6190406c575964db094fdd69d9fb387482a5ba2f19ed289b2194f698ce1',
    'launch_receipt.json':'7012ee94d666085f2c48256949a94f34392ccac6388e2ee928248c383ad536c0',
    'run_recovery_durable.ps1':'f89670e1bd4749ae7b8f85a3d124235b5c2259a49d6e7e6f05d8f39945bcb836',
    'verify_recovery_completed.py':'f06fb5e730cdb3416c058719f91d27ad17ccc07784538371cd6f72c500e263b8',
}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def subject(p):return dict(path=Path(p).as_posix(),sha256=sha(p))
def canonical(p):return str(Path(p).resolve()).replace('\\','/').casefold()
def canonical_map(values):
    out={}
    for path,digest in values.items():
        key=canonical(path)
        assert key not in out,('duplicate pin',path)
        out[key]=(path,digest)
    return out


def main():
    for name,digest in EXPECTED.items():assert sha(RUN/name)==digest,name
    request=read(RUN/'execution_request.json');frozen=read(RUN/'frozen_inputs.json');launch=read(RUN/'launch_receipt.json')
    prep=read(RUN/'source_preparation_v2.json');source_review=read(BASE/'source_review.json');boundary=read(BASE/'boundary_review.json')
    assert source_review['source_review_pass'] and boundary['boundary_review_pass']
    assert sha(BASE/'source_review.json')=='771a5ec79b84039d805533cd09446f1eb8c32370ec0b148e0c52919af05be28f'
    assert sha(BASE/'boundary_review.json')=='20bb1e718966c902c121e0812b1eb4f0ea28dea8601ae5cf172425c32247d06e'
    assert request['root_selected'] is True and request['source_only'] is False
    assert request['actual_input_preparation_run'] is True and request['actual_recovery_run'] is False
    assert request['dispatch_requires_separate_exact_root_concrete_clearance'] is True
    assert request['protocol']==frozen['protocol']==prep['protocol']==source_review['protocol']
    assert request['protocol']['labels_admissible'] is False
    assert frozen['request_sha256']==launch['request_sha256']==EXPECTED['execution_request.json']
    assert launch['frozen_receipt_sha256']==EXPECTED['frozen_inputs.json']
    assert launch['launcher_sha256']==EXPECTED['run_recovery_durable.ps1']
    assert launch['dispatch_authorized'] is False and launch['root_concrete_clearance_required'] is True
    assert frozen['source_sha256']==prep['source_sha256']==source_review['source_sha256'] and len(frozen['source_sha256'])==20
    frozen_pins=canonical_map(frozen['input_sha256']);launch_pins=canonical_map(launch['input_sha256'])
    assert len(frozen_pins)==38 and len(launch_pins)==74
    for key,(_,digest) in frozen_pins.items():assert launch_pins[key][1]==digest
    for role,item in request['subjects'].items():
        assert frozen_pins[canonical(item['path'])][1]==item['sha256'],role
    for role,path in {'selected_snapshot':RUN/'inputs/precontrol251.npz','selected_prefix':RUN/'inputs/actual_prefix251.npz',
                      'input_selection':RUN/'inputs/selection_receipt.json','source_review':BASE/'source_review.json',
                      'boundary_review':BASE/'boundary_review.json'}.items():
        item=request['subjects'][role]
        assert canonical(item['path'])==canonical(path) and item['sha256']==sha(path),role
    for name,digest in frozen['source_sha256'].items():
        path=RUN/'source_snapshot_v1'/name
        assert launch_pins[canonical(path)][1]==sha(path)==digest,name
    for path,digest in launch_pins.values():assert sha(path)==digest,path
    linux='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v1'
    mount='/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
    args=['-d','Ubuntu-22.04','--cd','/','--','bash',mount,'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
          'PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+linux+'/source_snapshot_v1','bash',linux+'/run_recovery.sh']
    assert launch['wsl_arguments']==args and all('\\' not in x for x in args)
    assert request['environment']['PYTHONPATH']==linux+'/source_snapshot_v1'
    interpreter='/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python'
    assert request['environment']['python']==interpreter
    shell=(RUN/'run_recovery.sh').read_bytes();assert b'\r' not in shell
    text=shell.decode();assert text.count('\nexec ')==1
    assert '\nbase='+linux+'\n' in text and '\nexec '+interpreter+' -B -u "$base/source_snapshot_v1/run_width251_actual_oracle.py"\n' in text
    assert text.index('linux_process.json')<text.index('\nexec ')
    assert read(RUN/'launcher_parse.json')['passed'] and read(RUN/'bash_parse_v2.json')['passed']
    assert read(RUN/'input_preparation_exit.json')['exit_code']==0
    tests=ET.parse(BASE/'concrete_helper_tests_final.xml').getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in tests)==25
    assert all(int(s.attrib.get(k,0))==0 for s in tests for k in ('failures','errors','skipped'))
    absent=['recovery_process_v1','execution_clearance.json','ATTEMPT_STARTED','initial_seed','nominal','post_lifecycle_hold_5s',
            'work_counters.json','driver_completion.json','outcome.json','failure.json']
    for name in absent:assert not (RUN/name).exists(),name
    helper_names=['prepare_execution_packet.py','amend_owner_packet_v2.py','verify_recovery_completed.py','run_recovery_durable.ps1',
                  'run_recovery.sh','test_execution_helpers.py']
    report=dict(passed=True,concrete_review_pass=True,review_scope='Source, saved boundary, exact launch metadata and synthetic process accounting only',
        request_sha256=EXPECTED['execution_request.json'],frozen_receipt_sha256=EXPECTED['frozen_inputs.json'],
        launch_receipt_sha256=EXPECTED['launch_receipt.json'],launcher_sha256=EXPECTED['run_recovery_durable.ps1'],
        request_subject=subject(RUN/'execution_request.json'),frozen_subject=subject(RUN/'frozen_inputs.json'),
        launch_receipt_subject=subject(RUN/'launch_receipt.json'),source_review_subject=subject(BASE/'source_review.json'),
        boundary_review_subject=subject(BASE/'boundary_review.json'),concrete_preparation_subject=subject(RUN/'concrete_preparation_v2.json'),
        source_sha256=frozen['source_sha256'],helper_sha256={name:sha(RUN/name) for name in helper_names},
        input_sha256=launch['input_sha256'],frozen_input_count=38,source_count=20,launch_pin_count=74,
        actual_role_count=len(request['subjects']),wsl_arguments=args,
        intended_wrapper_host='Windows PowerShell5.1 with -NoProfile -NonInteractive, hidden window and captured process handle',
        tests=dict(passed=True,count=25,independent_receipt_corruptions=8,producer_tests_independently_rerun=17,
                   receipt=subject(BASE/'concrete_helper_tests_final.xml')),
        required_clearance_argument='-ClearanceSha256 actual root clearance SHA256',
        processes_and_outputs_absent=absent,
        accounting=dict(exact_request_frozen_launch_review_chain=True,absence_start_child_dispatch_linux_receipts_hashed=True,
                        exact_work_categories_limits_and_integer_types=True,partial_primitive_progress_unknown=True,
                        expected_rejected_seed_not_automatically_fatal=True,raw_exit_driver_ledger_consistency=True,
                        completion_main1318_plus_conditional250=True,copied_prefix2510_steps_excluded=True),
        actual_recovery_dispatched=False,actual_recovery_execution_cleared=False,labels_admissible=False,
        task_model_calls=0,native_calls=0,replans=0,optimizer_calls=0,
        limitations=['Root exact concrete selection and clearance remain required before the sole expert-owned launch.',
                     'This source/packet review does not qualify recovered physics, intent, real-time performance or labels.',
                     'Missing final ledger or interrupted primitive calls retain accounting uncertainty; no automatic retry.',
                     'An earlier review test command named a nonexistent test file: no tests collected; final actual25 test capture passed.'],
        review_source=subject(Path(__file__)),independent_test_source=subject(BASE/'test_owner_receipts.py'))
    out=BASE/'concrete_review.json'
    with out.open('x',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(review=subject(out),passed=True,launch_pins=74,tests=25)))


if __name__=='__main__':main()
