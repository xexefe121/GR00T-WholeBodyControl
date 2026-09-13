"""Verify preserved ordinary-final evidence after the selected parity gate failed."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def main():
    fit=BASE/'fit';report=read(fit/'report.json');failure=read(fit/'failure.json');exit_report=read(BASE/'fit_process/exit.json')
    assert report['optimization_completed'] is True and report['final_export_diagnostics_completed'] is True
    assert report['completed'] is False and report['numerical_gate_passed'] is False and report['export_parity_passed'] is False
    assert report['ordinary_final_step']==55000 and report['additional_updates']==50000
    assert exit_report['raw_python_exit_code']==1 and exit_report['exit_code']==1 and exit_report['all_postrun_pins_exact'] is True
    assert failure['ordinary_final_preserved'] is True and failure['preservation_errors']==[] and failure['state']['optimizer_step_counters']==[55000]*6
    assert failure['state']['committed_loss_rows']==50000 and failure['automatic_retry_allowed'] is False
    expected=dict(training_head_rows_attempted=705500000,training_head_rows_returned=705500000,diagnostic_torch_rows_attempted=460740,diagnostic_torch_rows_returned=460740,ORT_calls_attempted=601,ORT_calls_returned=601)
    assert report['counters']==failure['counters']==expected
    assert sha(fit/'student_head.pt')==report['checkpoint_sha256'] and sha(fit/'student_head.onnx')==report['onnx_sha256']
    assert read(fit/'restoration.json')['all_restoration_checks_passed'] is True
    receipt=read(BASE/'training_frozen_inputs.json');pins=dict(receipt['input_sha256']);pins.update({str(Path(receipt['source_directory'])/k):v for k,v in receipt['source_sha256'].items()})
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed input '+path)
    outputs={p.relative_to(BASE).as_posix():sha(p) for p in sorted(fit.iterdir()) if p.is_file()}
    result=dict(owner_evidence_verification_passed=True,optimization_completed=True,numerical_gate_passed=False,canonical_evaluation_cleared=False,
        report_sha256=sha(fit/'report.json'),checkpoint_sha256=report['checkpoint_sha256'],onnx_sha256=report['onnx_sha256'],exit_sha256=sha(BASE/'fit_process/exit.json'),
        failure_sha256=sha(fit/'failure.json'),request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        clearance_sha256=sha(BASE/'training_clearance.json'),restoration_sha256=sha(fit/'restoration.json'),counters=expected,parity=report['export_parity'],output_sha256=outputs,
        source_sha256=sha(Path(__file__)),all_frozen_pins_exact=True,automatic_retry=False,task_model_calls=0,optimizer_updates=0,native_steps=0)
    with (BASE/'owner_failure_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(owner_evidence_verification_passed=True,numerical_gate_passed=False,sha256=sha(BASE/'owner_failure_verification.json'))))
if __name__=='__main__':main()
