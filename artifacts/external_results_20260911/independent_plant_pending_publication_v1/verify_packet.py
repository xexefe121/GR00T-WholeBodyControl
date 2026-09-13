"""Independent metadata inventory checks, no runner import or process dispatch."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def subject(p):return {'path':Path(p).resolve().as_posix(),'sha256':sha(p)}
def norm(p):return Path(p).resolve().as_posix()

def main():
    qp=BASE/'clock_request.json';lp=BASE/'clock_process/launch_receipt.json'
    assert sha(qp)=='2b538b03eb25de6d4c2ea3165561371af2415089957f4b9dd0bf9020a893118c'
    assert sha(lp)=='c7cc875c5f4c1d36e9efef5a56c8091d638e66d48393a0086bb2b30681c4dc3e'
    q,l=read(qp),read(lp);assert l['request_sha256']==sha(qp)
    assert q['execution_selected'] is False and l['execution_selected'] is False
    pins=l['input_hashes'];assert len(pins)==3768
    actual={p:sha(p) for p in pins};assert actual==pins
    request_pins={norm(e['path']):e['sha256'] for e in q['input_files']}
    assert len(request_pins)==len(q['input_files'])==3765
    assert all(pins[p]==h for p,h in request_pins.items())
    assert all(norm(p) in request_pins for p in q['roles'].values())
    original_root=NEW/'independent_plant_process_clock_v1'
    original=read(original_root/'clock_request.json');old_launch=read(original_root/'clock_process/launch_receipt.json')
    kept={norm(p):h for p,h in old_launch['input_hashes'].items() if not Path(p).resolve().is_relative_to(original_root.resolve())}
    assert len(old_launch['input_hashes'])==3718 and len(kept)==3689
    assert all(request_pins[p]==h for p,h in kept.items())
    allowed={'run_id','input_epoch','source_directory','output_directory','stage_directory','watchdog_budgets',
        'outer_process_timeout_seconds','job_publication_retry_contract','job_publication_retry_details',
        'execution_selected','preparation_only','execution_requires_separate_root_clearance',
        'byte_preserved_copies','roles','input_files','prior_failed_attempt'}
    changed={k for k in set(q)|set(original) if q.get(k)!=original.get(k)}
    assert changed<=allowed
    assert all(q['roles'][k]==v for k,v in original['roles'].items())
    assert q['input_epoch']==3 and q['job_publication_retry_contract']=='pending_BUSY_same_job_max10_before_original_activation'
    assert q['requested_controls']==1819 and q['main_controls']==1569 and q['hold_controls']==250
    assert q['native_step_budget']==18190 and q['serialization_budget']==4
    assert q['epoch_lead_ns']==200000000 and q['debt_abort_steps']==100 and q['elapsed_abort_ns']==60000000000
    assert q['watchdog_budgets']==dict(setup_ns=240000000000,plant_ns=120000000000,preservation_ns=180000000000,outer_timeout_seconds=555,outer_kill_grace_seconds=5)
    assert q['outer_process_timeout_seconds']==555 and q['epoch_rebase_allowed'] is False
    producer_review=read(q['roles']['pending_source_review'])
    assert sha(q['roles']['pending_source_review'])=='d2f303a488817a1f088790fbe37a8d411576ddfb7b84c1043c5309dbc7a4e357'
    assert producer_review['passed'] is True
    source_map={p.name:sha(p) for p in (BASE/'source_draft_v1').glob('*.py')}
    assert len(source_map)==22 and source_map==producer_review['source_sha256']
    ar=Path(q['roles']['saved_stage_audit_source_review']);audit_review=read(ar)
    assert sha(ar)=='81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0'
    audit_source=NEW/'independent_plant_pending_publication_saved_audit_v1/source_draft_v2'
    assert audit_review['source_sha256']=={p.name:sha(p) for p in audit_source.glob('*.py')}
    assert audit_review['required_clock_request_contract']==q['job_publication_retry_contract']
    assert audit_review['pending_core_sha256']==source_map['clock_core.py']
    for name,record in q['byte_preserved_copies'].items():
        assert sha(record['original_path'])==record['sha256']==source_map[name]
    args=l['exact_wsl_arguments']
    assert args[:6]==['-d','Ubuntu-22.04','--cd','/','--','bash']
    assert args[args.index('timeout')+1:args.index('timeout')+4]==['--signal=TERM','--kill-after=5s','555s']
    assert args[-1].endswith('/independent_plant_pending_publication_v1/clock_process/launch_clearance.json')
    parse=read(BASE/'concrete_launcher_parse.json')
    assert parse['passed'] is True and parse['normalization_passed'] is True
    assert parse['literal_separator_length']==1 and parse['literal_separator_codepoint']==92
    for name,preview in [('run.ps1','preview_run.ps1.txt'),('run_durable.ps1','preview_durable.ps1.txt')]:
        p=BASE/'clock_process'/name
        assert p.read_bytes()==(BASE/preview).read_bytes()
        assert any(v['path']==norm(p) and v['sha256']==sha(p) and v['passed'] is True for v in parse['launchers'])
    absent=[BASE/p for p in ('clock_process/launch_clearance.json','clock_process/started.lock','clock_process/start.json','clock_process/child.json',
        'clock_process/exit.json','clock_process/stdout.log','clock_process/stderr.log','run','stage_receipts')]
    assert all(not p.exists() for p in absent)
    saved=NEW/'independent_plant_pending_publication_saved_actual_v1'
    saved_prep=read(saved/'helper_preparation.json');assert saved_prep['passed'] is True
    for name,digest in saved_prep['helper_sha256'].items():assert sha(saved/name)==digest
    for name,digest in saved_prep['evidence_sha256'].items():assert sha(saved/name)==digest
    assert saved_prep['audit_source_sha256']==audit_review['source_sha256']
    assert saved_prep['audit_source_review_sha256']==sha(ar)
    for name in ('request.json','launch_receipt.json','launch_clearance.json','process_v1','results_v1','dispatch.json'):
        assert not (saved/name).exists()
    paths=[qp,lp,BASE/'helper_preparation.json',BASE/'input_scope_derivation.json',BASE/'check_concrete.ps1',BASE/'concrete_launcher_parse.json',Path(__file__),
        saved/'helper_preparation.json',saved/'README.md',ar]
    report=dict(passed=True,preparation_only=True,subjects={norm(p):subject(p) for p in paths},
        input_sha256=actual,request_pin_count=len(request_pins),launch_pin_count=len(pins),original_external_pins_preserved=len(kept),
        original_roles_unchanged=True,only_allowed_request_changes=sorted(changed),producer_source_sha256=source_map,
        saved_auditor_source_sha256=audit_review['source_sha256'],helper_sha256=read(BASE/'helper_preparation.json')['helper_sha256'],
        saved_audit_helper_sha256=saved_prep['helper_sha256'],exact_wsl_arguments=args,absent_paths=[norm(p) for p in absent],
        watchdog_budgets=q['watchdog_budgets'],clock_selected=False,saved_audit_request_created=False,clearance_created=False,
        actual_dispatch=False,native_steps=0,model_calls=0,optimizer_updates=0,MJB_serializations=0,worker_processes=0)
    out=BASE/'concrete_metadata_preparation.json'
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,report_sha256=sha(out),request_sha256=sha(qp),launch_receipt_sha256=sha(lp),pins=len(pins),dispatched=False)))

if __name__=='__main__':main()
