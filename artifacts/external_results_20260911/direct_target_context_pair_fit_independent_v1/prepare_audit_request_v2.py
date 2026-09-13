"""Freeze a completed paired fit for one separately selected saved audit. No forwards."""
import argparse,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
EXPERIMENT=BASE.parent/'direct_target_causal_context_study_v1'
PYTHON='E:/codex_sonic_runtime/direct_target_gpu_20260911/venv/Scripts/python.exe'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main(args):
    review=read(args.source_review)
    assert sha(args.source_review)==args.source_review_sha256 and review[args.pass_field] is True
    source=BASE/'source_draft_v2';sources={p.resolve().as_posix():sha(p) for p in sorted(source.glob('*.py'))}
    for p,d in sources.items():assert review['source_sha256'][Path(p).name]==d
    fit=EXPERIMENT/'fit';shared=fit/'shared';pair_path=fit/'paired_report.json'
    assert sha(pair_path)==args.paired_report_sha256
    pair=read(pair_path);assert pair['completed'] is True and pair['conditions']==['blinded','causal']
    owner_path=EXPERIMENT/'owner_completion_verification_v3.json'
    assert sha(owner_path)==args.owner_sha256
    owner=read(owner_path);assert owner['owner_verification_passed'] is True and owner['paired_completion_passed'] is True and owner['processes_absent'] is True
    request=read(EXPERIMENT/'training_request.json')
    roles=dict(training_request=EXPERIMENT/'training_request.json',frozen_inputs=EXPERIMENT/'training_frozen_inputs.json',
        training_manifest=EXPERIMENT/'training_frozen_inputs.json',paired_report=pair_path,owner_completion=owner_path,
        normalization=shared/'normalization.npz',shared_manifest=shared/'output_manifest.json',context_alignment=shared/'context_alignment.json',
        coefficient=Path(request['subjects']['coefficient_source']['path']),source_checkpoint=Path(request['subjects']['checkpoint']['path']),
        full_state_generation_request=Path(request['full_state_paths']['request']),full_state_generation_report=Path(request['full_state_paths']['report']),
        full_state_data_audit=Path(request['subjects']['full_state_root_audit']['path']),full_state_data_owner=Path(request['subjects']['full_state_root_owner']['path']))
    for condition in ('blinded','causal'):
        for role,name in [('fit_report','report.json'),('checkpoint','student_head.pt'),('head','student_head.onnx'),('export_manifest','output_manifest.json')]:roles[condition+'_'+role]=fit/condition/name
        roles[condition+'_owner_completion']=EXPERIMENT/(condition+'_owner_completion_verification.json')
        assert sha(fit/condition/'report.json')==pair['condition_report_sha256'][condition]
    result=dict(kind='saved_matched_context_pair_only',experiment=EXPERIMENT.as_posix(),python_path=PYTHON,
        source_review=dict(path=args.source_review.resolve().as_posix(),sha256=args.source_review_sha256,pass_field=args.pass_field),
        source_sha256=sources,subjects={r:dict(path=p.resolve().as_posix(),sha256=sha(p)) for r,p in roles.items()},
        requires_completed_pair=True,automatic_retry=False,task_model_calls=0,ORT_calls=0,gradient_calls=0,native_steps=0,optimizer_updates=0,
        request_writer_sha256=sha(__file__))
    with (BASE/'audit_request.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(request_path=(BASE/'audit_request.json').as_posix(),request_sha256=sha(BASE/'audit_request.json'))))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-review',type=Path,required=True);p.add_argument('--source-review-sha256',required=True)
    p.add_argument('--pass-field',default='source_review_pass');p.add_argument('--paired-report-sha256',required=True);p.add_argument('--owner-sha256',required=True)
    main(p.parse_args())
