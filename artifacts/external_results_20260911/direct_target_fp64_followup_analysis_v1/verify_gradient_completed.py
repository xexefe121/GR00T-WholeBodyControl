"""Saved-only owner verification; no model, optimizer or native imports."""
from pathlib import Path
import hashlib
import json

BASE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    request=read(BASE/'gradient_request.json')
    report=read(BASE/'gradient_results/report.json')
    exit_report=read(BASE/'gradient_process_v2/exit.json')
    launch=read(BASE/'gradient_process_v2/launch.json')
    post=read(BASE/'gradient_process_v2/postrun_sha256.json')
    absence=read(BASE/'gradient_process_absence.json')
    failed_launch=read(BASE/'gradient_process/exit.json')
    root=BASE.parent/'direct_target_gradient_saved_review_v1'
    root_report=read(root/'report.json')
    assert report['passed'] and report['parameter_state_unchanged']
    assert report['request_sha256']==sha(BASE/'gradient_request.json')
    assert report['clearance_sha256']==sha(BASE/'gradient_clearance.json')
    assert report['counters']==dict(forward_attempted=3,forward_returned=3,forward_rows_attempted=14110,
        forward_rows_returned=14110,gradient_attempted=3,gradient_returned=3)
    assert report['optimizer_updates']==0 and report['native_calls']==0
    assert exit_report['raw_exit_known'] and exit_report['raw_python_exit_code']==0 and exit_report['exit_code']==0
    assert exit_report['wrapper_error'] is None
    assert exit_report['wrapper_pid']==launch['wrapper_pid'] and exit_report['child_pid']==launch['child_pid']
    assert absence['all_absent'] and absence['present_pids']==[]
    assert set(absence['expected_pids'])=={exit_report['wrapper_pid'],exit_report['child_pid'],failed_launch['wrapper_pid']}
    assert failed_launch['child_pid'] is None and failed_launch['exit_code']==99
    for path,digest in request['input_sha256'].items():
        assert post[path]==digest and sha(path)==digest
    for name,digest in report['output_sha256'].items():assert sha(BASE/'gradient_results'/name)==digest
    assert sha(BASE/'gradient_diagnostic.py')==request['source_sha256']==launch['source_sha256']
    assert root_report['saved_gradient_math_passed']
    for path,digest in root_report['input_sha256'].items():assert sha(path)==digest
    subjects={key:dict(path=path.as_posix(),sha256=sha(path)) for key,path in dict(
        source=BASE/'gradient_diagnostic.py',request=BASE/'gradient_request.json',frozen=BASE/'gradient_frozen_inputs.json',
        clearance=BASE/'gradient_clearance.json',report=BASE/'gradient_results/report.json',exit=BASE/'gradient_process_v2/exit.json',
        launcher=BASE/'run_gradient_durable_v2.ps1',postpins=BASE/'gradient_process_v2/postrun_sha256.json',
        absence=BASE/'gradient_process_absence.json',prior_exit=BASE/'gradient_process/exit.json',
        repair=BASE/'gradient_launcher_repair.json',source_review=root/'source_review.json',independent_saved_math=root/'report.json').items()}
    result=dict(completed=True,passed=True,subjects=subjects,all_input_output_hashes_exact=True,
        counters=report['counters'],optimizer_updates=0,native_calls=0,original_failed_FP32_and_simulation_preserved=True,
        first_launcher_failed_before_Python=True,one_task_execution=True,processes_absent=True,
        source_sha256=sha(__file__),selected_training_weight=None)
    output=BASE/'gradient_owner_completion.json'
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,owner_sha256=sha(output),model_calls=0)))

if __name__=='__main__':main()
