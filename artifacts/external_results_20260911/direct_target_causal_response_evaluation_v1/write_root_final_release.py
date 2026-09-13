"""Bind a completed independent audit and fixed ordinary71000 endpoint; no inference."""
from pathlib import Path
import argparse,hashlib,json,datetime
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit-directory',type=Path,required=True)
    p.add_argument('--audit-sha256',required=True)
    p.add_argument('--audit-owner-sha256',required=True)
    p.add_argument('--helper-review',type=Path,required=True)
    p.add_argument('--helper-review-sha256',required=True)
    a=p.parse_args();audit_path=a.audit_directory/'results_v1/report.json';owner_path=a.audit_directory/'owner_completion.json'
    assert sha(audit_path)==a.audit_sha256 and sha(owner_path)==a.audit_owner_sha256
    audit=read(audit_path);owner=read(owner_path);request=read(a.audit_directory/'audit_request.json')
    for k in ('completion_accounting_passed','evidence_audit_passed','export_qualified','processes_absent','raw_exit_known','all_pre_post_current_launch_pins_exact'):assert owner[k] is True,k
    assert owner['raw_python_exit_code']==owner['exit_code']==0
    assert owner['report_sha256']==a.audit_sha256 and owner['request_sha256']==sha(a.audit_directory/'audit_request.json')
    assert owner['output_sha256'][audit_path.resolve().as_posix()]==a.audit_sha256
    assert audit['evidence_audit_passed'] is audit['export_qualified'] is True
    assert audit['audit_request_sha256']==owner['request_sha256']
    for k in ('model_calls','ORT_calls','gradient_calls','optimizer_updates','native_steps'):assert audit[k]==0
    subjects={k:v for k,v in request['subjects'].items() if k not in ('frozen_inputs','owner_completion')}
    assert len(subjects)==16
    for k,v in subjects.items():
        assert sha(v['path'])==v['sha256']==audit['direct_subject_sha256'][k]
        assert audit['input_sha256'][Path(v['path']).resolve().as_posix()]==v['sha256']
    fit_owner=Path(request['subjects']['owner_completion']['path'])
    assert sha(fit_owner)==request['subjects']['owner_completion']['sha256']=='a6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3'
    fit=read(subjects['fit_report']['path'])
    assert fit['ordinary_final_step']==71000 and fit['optimization_completed'] is fit['numerical_gate_passed'] is True
    assert subjects['fit_report']['sha256']=='f918f1dfe675aae01e25307d302ce6c54f599fb7f0f5d1fa3fd943a179b5aded'
    assert subjects['checkpoint']['sha256']=='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d'
    assert subjects['head']['sha256']=='c70e90adab31efd2abcebd9817bf3c3c1ecc84cde546e080f6367c710633423a'
    source_path=NEW/'direct_target_causal_response_evaluation_independent_review_v1/review.json'
    assert sha(source_path)=='e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121'
    source=read(source_path);assert source['passed'] is True
    for name,digest in source['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest
    assert sha(a.helper_review)==a.helper_review_sha256
    helper=read(a.helper_review);assert helper['passed'] is True
    assert helper['helper_sha256']==read(BASE/'launch_helper_preparation_v2.json')['helper_sha256']
    for name,digest in helper['helper_sha256'].items():assert sha(BASE/name)==digest
    direct={k:v['sha256'] for k,v in subjects.items()}
    direct.update(root_training_audit=a.audit_sha256,fit_owner_completion=sha(fit_owner))
    release=dict(release_review_pass=True,reviewer='root',reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),context_condition='causal',ordinary_final_step=71000,direct_subject_sha256=direct,subjects=subjects,audit_path=audit_path.as_posix(),audit_owner_sha256=a.audit_owner_sha256,source_review_sha256=sha(source_path),helper_review_sha256=a.helper_review_sha256,root_selection_reason='Fixed ordinary71000 endpoint reduces saved nominal, balanced response and physical losses. Response still exceeds zero-response baseline. Select one full original canonical simulation to measure stability, with no numerical or physical gate relaxation.',same_source_weights_FP64_export=True,physical_qualification=False,selected_witness_calls=1,selected_main_controls=1569,conditional_hold_controls=250,per_stage_concrete_review_required=True,native_steps=0,model_calls=0,scope='One original canonical simulation pipeline; full same-controller acceptance remains pending.',writer_sha256=sha(__file__))
    with (BASE/'root_final_release_review.json').open('x',encoding='utf-8') as f:json.dump(release,f,indent=2);f.write('\n')
    print(json.dumps({'release_review_pass':True,'sha256':sha(BASE/'root_final_release_review.json'),'release_roles':16}))
if __name__=='__main__':main()
