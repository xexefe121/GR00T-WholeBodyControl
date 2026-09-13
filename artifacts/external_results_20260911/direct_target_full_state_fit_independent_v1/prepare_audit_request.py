"""Freeze actual saved-fit subjects after a completed fit; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
EXPERIMENT=BASE.parent/'direct_target_full_state_student_v1'
PYTHON='E:/codex_sonic_runtime/direct_target_gpu_20260911/venv/Scripts/python.exe'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main(args):
    fit=EXPERIMENT/'fit';report_path=fit/'report.json'
    assert sha(report_path)==args.fit_report_sha256
    report=read(report_path)
    assert report['optimization_completed'] is True and report['final_export_diagnostics_completed'] is True
    assert report['ordinary_final_step']==65000 and report['optimizer_step']==10000
    source_review=read(args.source_review);assert sha(args.source_review)==args.source_review_sha256 and source_review[args.pass_field] is True
    source=BASE/'source_draft_v1';sources={str(p.resolve()):sha(p) for p in sorted(source.glob('*.py'))}
    for p,d in sources.items():assert source_review['source_sha256'][Path(p).name]==d
    request=read(EXPERIMENT/'training_request.json')
    paths=dict(fit_report=report_path,checkpoint=fit/'student_head.pt',head=fit/'student_head.onnx',normalization=fit/'normalization.npz',
        training_manifest=EXPERIMENT/'training_frozen_inputs.json',training_request=EXPERIMENT/'training_request.json',export_manifest=fit/'output_manifest.json',coefficient=fit/'coefficient.json',
        source_checkpoint=Path(request['subjects']['checkpoint']['path']),full_state_generation_request=Path(request['full_state_paths']['request']),
        full_state_generation_report=Path(request['full_state_paths']['report']),full_state_data_audit=Path(request['subjects']['full_state_root_audit']['path']),
        full_state_data_owner=Path(request['subjects']['full_state_root_owner']['path']))
    assert sha(paths['training_request'])=='0c3bbc4e8567f6dfc09dfe7d9cdfcdd8af5a56178ea112af3a647d84c48d0dea'
    assert sha(paths['training_manifest'])=='bbb67b48187dcead0dc0886496573447ffdd58582eaeb63f41a7fa947e922f18'
    exit_record=read(EXPERIMENT/'fit_process/exit.json')
    assert exit_record['exit_known'] is True and exit_record['all_postrun_pins_exact'] is True
    result=dict(kind='saved_full58_fit_evidence_only',experiment=EXPERIMENT.as_posix(),python_path=PYTHON,
        source_review=dict(path=args.source_review.resolve().as_posix(),sha256=args.source_review_sha256,pass_field=args.pass_field),
        source_sha256=sources,subjects={role:dict(path=path.resolve().as_posix(),sha256=sha(path)) for role,path in paths.items()},
        expected_optimization_completed=True,allow_preserved_numerical_failure=True,task_model_calls=0,native_steps=0,optimizer_updates=0,
        request_writer_sha256=sha(__file__))
    with (BASE/'audit_request.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print((BASE/'audit_request.json').as_posix(),sha(BASE/'audit_request.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-review',type=Path,required=True);p.add_argument('--source-review-sha256',required=True)
    p.add_argument('--pass-field',default='source_review_pass');p.add_argument('--fit-report-sha256',required=True)
    main(p.parse_args())
