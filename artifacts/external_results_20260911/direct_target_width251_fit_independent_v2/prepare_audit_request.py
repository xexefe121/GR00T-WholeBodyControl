"""Future completed saved-fit binding only. Never starts the auditor or a model."""
import argparse,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
EXPERIMENT=BASE.parent/'direct_target_width251_student_v1'
PYTHON='E:/codex_sonic_runtime/direct_target_gpu_20260911/venv/Scripts/python.exe'
sys.path.insert(0,str(BASE/'source_prepared_v1'))
from audit_release import release_paths

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main(args):
    review=read(args.source_review)
    assert sha(args.source_review)==args.source_review_sha256 and review[args.pass_field] is True
    assert review['helper_sha256']['prepare_audit_request.py']==sha(__file__)
    source=BASE/'source_prepared_v1';sources={p.resolve().as_posix():sha(p) for p in sorted(source.glob('*.py'))}
    for p,d in sources.items():assert review['source_sha256'][Path(p).name]==d
    fit=EXPERIMENT/'fit';report_path=fit/'report.json';owner_path=EXPERIMENT/'owner_completion_verification.json'
    assert sha(report_path)==args.fit_report_sha256 and sha(owner_path)==args.owner_sha256
    report=read(report_path);owner=read(owner_path)
    assert report['optimization_completed'] is True and report['final_export_diagnostics_completed'] is True
    assert report['ordinary_final_step']==91000 and report['optimizer_step']==26000 and report['hidden_width']==512
    assert owner['owner_verification_passed'] is True and owner['accounting_passed'] is True and owner['processes_absent'] is True
    request=read(EXPERIMENT/'training_request.json');roles=release_paths(EXPERIMENT,request)
    roles.update(frozen_inputs=EXPERIMENT/'training_frozen_inputs.json',owner_completion=owner_path)
    result=dict(kind='saved_width251_recovery_warm_only',experiment=EXPERIMENT.as_posix(),python_path=PYTHON,
        source_review=dict(path=args.source_review.resolve().as_posix(),sha256=args.source_review_sha256,pass_field=args.pass_field),
        source_sha256=sources,subjects={r:dict(path=p.resolve().as_posix(),sha256=sha(p)) for r,p in roles.items()},
        requires_completed_optimization_and_diagnostics=True,numerical_failure_retained_separately=True,
        automatic_retry=False,task_model_calls=0,ORT_calls=0,gradient_calls=0,native_steps=0,optimizer_updates=0,
        request_writer=dict(path=Path(__file__).resolve().as_posix(),sha256=sha(__file__)))
    with (BASE/'audit_request.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(request_path=(BASE/'audit_request.json').as_posix(),request_sha256=sha(BASE/'audit_request.json'))))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-review',type=Path,required=True);p.add_argument('--source-review-sha256',required=True)
    p.add_argument('--pass-field',default='source_review_pass');p.add_argument('--fit-report-sha256',required=True);p.add_argument('--owner-sha256',required=True)
    main(p.parse_args())
