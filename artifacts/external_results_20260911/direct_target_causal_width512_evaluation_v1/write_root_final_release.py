"""Bind a completed independent audit and fixed ordinary81000 endpoint; no inference."""
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
    assert sha(fit_owner)==request['subjects']['owner_completion']['sha256']=='18c66ebdb2af7138858c08d012c5d3cc7819c1bb2bb5a5b523e488c35e4ed6a2'
    fit=read(subjects['fit_report']['path'])
    assert fit['ordinary_final_step']==81000 and fit['optimization_completed'] is fit['numerical_gate_passed'] is True
    assert fit['hidden_width']==512 and fit['architecture']==[1323,512,512,23] and fit['optimizer_step']==16000
    assert subjects['fit_report']['sha256']=='b5979023fe9bceb14e0d54cf04bf727fbc413d5a1443b39018e149443b418de6'
    assert subjects['checkpoint']['sha256']=='825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e'
    assert subjects['head']['sha256']=='8a4c394b8836dd763d6eb1c5beac73bf49921bcd59fc5f04850b377d0e843044'
    source_path=NEW/'direct_target_width512_evaluation_independent_review_v2/review.json'
    assert sha(source_path)=='db2a997d44634cb59d7d4bd92408fcf361323d143c6109024b457e5bd4a458de'
    source=read(source_path);assert source['passed'] is True
    for name,digest in source['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest
    assert sha(a.helper_review)==a.helper_review_sha256
    helper=read(a.helper_review);assert helper['passed'] is True
    assert helper['helper_sha256']==read(BASE/'launch_helper_preparation_v2.json')['helper_sha256']
    for name,digest in helper['helper_sha256'].items():assert sha(BASE/name)==digest
    direct={k:v['sha256'] for k,v in subjects.items()}
    direct.update(root_training_audit=a.audit_sha256,fit_owner_completion=sha(fit_owner))
    release=dict(release_review_pass=True,reviewer='root',reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),context_condition='causal',ordinary_final_step=81000,direct_subject_sha256=direct,subjects=subjects,audit_path=audit_path.as_posix(),audit_owner_sha256=a.audit_owner_sha256,source_review_sha256=sha(source_path),helper_review_sha256=a.helper_review_sha256,root_selection_reason='Fixed ordinary81000 width512 endpoint reduces saved nominal, balanced response and physical losses after10000 updates. Balanced response still exceeds zero-response baseline. Select one full original canonical simulation to measure stability and actual runtime, with original numerical and physical gates.',same_source_weights_FP64_export=True,physical_qualification=False,selected_witness_calls=1,selected_main_controls=1569,conditional_hold_controls=250,per_stage_concrete_review_required=True,native_steps=0,model_calls=0,scope='One original canonical simulation pipeline; full same-controller acceptance remains pending.',writer_sha256=sha(__file__))
    with (BASE/'root_final_release_review.json').open('x',encoding='utf-8') as f:json.dump(release,f,indent=2);f.write('\n')
    print(json.dumps({'release_review_pass':True,'sha256':sha(BASE/'root_final_release_review.json'),'release_roles':16}))
if __name__=='__main__':main()
