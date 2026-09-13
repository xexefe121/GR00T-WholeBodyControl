"""Owner saved-artifact verification only; zero model/native/optimizer calls."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def main():
    fit=BASE/'fit';report=read(fit/'report.json');exit_report=read(BASE/'fit_process/exit.json')
    for key in ('completed','optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed'):
        if report[key] is not True:raise ValueError(key+' not passed')
    if exit_report['raw_python_exit_code']!=0 or exit_report['exit_code']!=0 or exit_report['error'] is not None:raise ValueError('Nonzero/unknown durable exit')
    if report['ordinary_final_step']!=55000 or report['additional_updates']!=50000:raise ValueError('Wrong final step')
    expected=dict(training_head_rows_attempted=705500000,training_head_rows_returned=705500000,diagnostic_torch_rows_attempted=460740,diagnostic_torch_rows_returned=460740,ORT_calls_attempted=601,ORT_calls_returned=601)
    if report['counters']!=expected:raise ValueError('Incorrect fixed counters')
    if sha(fit/'student_head.pt')!=report['checkpoint_sha256'] or sha(fit/'student_head.onnx')!=report['onnx_sha256']:raise ValueError('Final head identity mismatch')
    restoration=read(fit/'restoration.json')
    if restoration['all_restoration_checks_passed'] is not True:raise ValueError('Restoration did not pass')
    receipt=read(BASE/'training_frozen_inputs.json')
    pins=dict(receipt['input_sha256']);pins.update({str(Path(receipt['source_directory'])/k):v for k,v in receipt['source_sha256'].items()})
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed pin '+path)
    if read(BASE/'fit_process/postrun_pins.json')['all_exact'] is not True:raise ValueError('Launcher final pins did not pass')
    outputs={p.relative_to(BASE).as_posix():sha(p) for p in sorted(fit.iterdir()) if p.is_file()}
    result=dict(owner_verification_passed=True,report_sha256=sha(fit/'report.json'),checkpoint_sha256=report['checkpoint_sha256'],onnx_sha256=report['onnx_sha256'],
        exit_sha256=sha(BASE/'fit_process/exit.json'),request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        clearance_sha256=sha(BASE/'training_clearance.json'),restoration_sha256=sha(fit/'restoration.json'),counters=report['counters'],parity=report['export_parity'],
        summary={key:report[key] for key in ('initial_metrics','final_metrics')},output_sha256=outputs,task_model_calls=0,optimizer_updates=0,native_steps=0,behavioral_qualification=False)
    with (BASE/'owner_completion_verification.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(owner_verification_passed=True,sha256=sha(BASE/'owner_completion_verification.json'))))
if __name__=='__main__':main()
