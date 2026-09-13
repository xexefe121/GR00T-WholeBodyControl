"""Saved-only owner completion verification; no models or native calls."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    dest=BASE/'export';report=read(dest/'report.json');request=read(BASE/'export_request.json');receipt=read(BASE/'export_frozen_inputs.json');exit_report=read(BASE/'export_process/exit.json');absence=read(BASE/'process_absence.json')
    for key in ('completed','validation_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged'):
        if report[key] is not True:raise ValueError('Missing '+key)
    if exit_report['raw_python_exit_code']!=0 or exit_report['exit_code']!=0 or exit_report['exit_known'] is not True or exit_report['all_postrun_pins_exact'] is not True or absence['any_present'] is not False:raise ValueError('Durable process completion not verified')
    if report['parity_tolerance_rad']!=1e-5 or report['max_preclip_error_rad']>1e-5:raise ValueError('Numerical gate differs')
    if report['optimizer_updates']!=0 or report['BFM_calls']!=0 or report['native_steps']!=0 or report['trace_forward_calls']!=0:raise ValueError('Unexpected calls')
    for counters in report['counters'].values():
        if any(counters[k]!=601 for k in ('calls_attempted','calls_returned','calls_synchronized','calls_verified')) or any(counters[k]!=153580 for k in ('rows_attempted','rows_returned','rows_verified')):raise ValueError('Fixed call counts differ')
    for path,digest in receipt['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed input '+path)
    for name,digest in receipt['source_sha256'].items():
        if sha(Path(receipt['source_directory'])/name)!=digest:raise ValueError('Changed source '+name)
    if sha(dest/'manifest.json')!=report['manifest_sha256']:raise ValueError('Changed manifest')
    for entry in read(dest/'manifest.json')['files'].values():
        if sha(dest/entry['path'])!=entry['sha256']:raise ValueError('Changed output '+entry['path'])
    mapping={key:request['subjects'][key]['sha256'] for key in ('checkpoint','fit_report','source_head','normalization')}
    mapping.update(head=sha(dest/'student_head_fp64.onnx'),export_report=sha(dest/'report.json'),export_request=sha(BASE/'export_request.json'),export_manifest=sha(dest/'manifest.json'))
    if mapping['head']!=report['onnx_sha256']:raise ValueError('New head changed')
    result=dict(owner_verification_passed=True,completed=True,numerical_gate_passed=True,export_parity_passed=True,direct_subject_sha256=mapping,
        raw_exit_known=True,raw_exit_code=0,raw_python_exit_code=exit_report['raw_python_exit_code'],exit_code=0,all_postrun_pins_exact=True,processes_absent=True,exit_sha256=sha(BASE/'export_process/exit.json'),process_absence_sha256=sha(BASE/'process_absence.json'),
        frozen_receipt_sha256=sha(BASE/'export_frozen_inputs.json'),clearance_sha256=sha(BASE/'export_clearance.json'),source_sha256=sha(Path(__file__)),counters=report['counters'],parity=read(dest/'parity.json'),
        task_model_calls=0,optimizer_updates=0,native_steps=0,canonical_evaluation_qualified=False)
    with (BASE/'owner_completion_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(owner_verification_passed=True,sha256=sha(BASE/'owner_completion_verification.json'),direct_subject_sha256=mapping)))
if __name__=='__main__':main()
