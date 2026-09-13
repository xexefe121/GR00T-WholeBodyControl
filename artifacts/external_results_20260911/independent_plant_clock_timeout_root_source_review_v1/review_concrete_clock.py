"""Review exact corrected clock request and launcher without dispatching it."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

NEW = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE = NEW / 'independent_plant_clock_timeout_correction_v1'
OLD = NEW / 'independent_plant_process_clock_v1'
OUT = Path(__file__).resolve().parent / 'concrete_review.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

rp = BASE / 'clock_request.json'
lp = BASE / 'clock_process/launch_receipt.json'
assert sha(rp) == '0c22640dfb617a20c31aff7356434a6253096b0fc4cbed38aae64faeb14e75eb'
assert sha(lp) == 'f3eb4aca76e45b56761a95017090359a4de21d6edd1f3d7cacc71754158f1713'
r, l = read(rp), read(lp)
o, ol = read(OLD/'clock_request.json'), read(OLD/'clock_process/launch_receipt.json')
assert sha(OLD/'clock_request.json') == 'd6a6c3332039046c77ff9c93eaedd0694f0ac3c8487a474760c25646e75ba91a'
assert sha(OLD/'clock_process/launch_receipt.json') == '4ecd884ea49e2b307febb63967c222946c9ef6fdce73a8b52c59571b9a383c80'
unchanged = ('requested_controls','main_controls','hold_controls','native_step_budget','serialization_budget',
    'model_inference_calls','optimizer_updates','other_oracle_native_steps','epoch_lead_ns','epoch_rebase_allowed',
    'debt_abort_steps','elapsed_abort_ns','native_bundle','expected_model_sha256','command_table_sha256')
assert all(r[k] == o[k] for k in unchanged)
assert (r['requested_controls'], r['main_controls'], r['hold_controls'], r['native_step_budget'], r['serialization_budget']) == (1819,1569,250,18190,4)
assert r['model_inference_calls'] == r['optimizer_updates'] == r['other_oracle_native_steps'] == 0
assert r['epoch_lead_ns'] == 200000000 and r['epoch_rebase_allowed'] is False
assert r['debt_abort_steps'] == 100 and r['elapsed_abort_ns'] == 60000000000
assert r['watchdog_budgets'] == dict(setup_ns=240000000000,plant_ns=120000000000,preservation_ns=180000000000,outer_timeout_seconds=555,outer_kill_grace_seconds=5)
assert r['outer_process_timeout_seconds'] == 555
assert r['hardware_authorized'] is False and l['hardware_authorized'] is False
assert r['execution_selected'] is False and l['execution_selected'] is False
assert l['automatic_retry'] is False and l['final_clearance_required'] is True
assert l['request_path'] == rp.as_posix() and l['request_sha256'] == sha(rp)
assert l['requested_native_steps'] == 18190 and l['requested_serializations'] == 4
assert all(r['roles'][k] == v for k,v in o['roles'].items())
assert Path(r['output_directory']).resolve() == (BASE/'run').resolve()
assert Path(r['stage_directory']).resolve() == (BASE/'stage_receipts').resolve()
assert Path(r['source_directory']).resolve() == (BASE/'source_draft_v1').resolve()
assert r['input_epoch'] == 2 and r['run_id'] == 'query250-recorded-clock-timeout-v2'
pins=l['input_hashes']
assert len(pins) == 3745 and len(r['input_files']) == 3742
for item in r['input_files']:
    assert pins[item['path']] == item['sha256']
for p,h in pins.items():
    assert sha(p) == h, p
external=0
for p,h in ol['input_hashes'].items():
    if not Path(p).resolve().is_relative_to(OLD.resolve()):
        assert pins[p] == h
        external+=1
for role, expected in (
    ('timeout_source_review','01939a4bf82d4f1b2e19e7240ea0f4aa4e6eba49c7dac71116a1c24b509d13c2'),
    ('saved_stage_audit_source_review','278c509418ef1c7a1be655693a840217dd8c1894a6a6c33f0757a712ad8b9ac7')):
    assert pins[r['roles'][role]] == expected and read(r['roles'][role])['passed'] is True
prep = read(BASE/'source_preparation_v1.json')
assert read(r['roles']['timeout_source_review'])['source_sha256'] == prep['source_sha256']
for p,h in prep['source_sha256'].items():
    assert sha(BASE/p) == h
ap = read(r['roles']['saved_stage_audit_preparation'])
assert read(r['roles']['saved_stage_audit_source_review'])['source_sha256'] == ap['source_sha256']
for p,h in ap['source_sha256'].items():
    assert sha(NEW/'independent_plant_clock_saved_root_review_v1/source_audit_v3'/p) == h
linux='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/'+BASE.name
args=['-d','Ubuntu-22.04','--cd','/','--','bash',
 '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
 'timeout','--signal=TERM','--kill-after=5s','555s','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
 'MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+linux+'/source_draft_v1',
 '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python','-B',linux+'/source_draft_v1/run_clock.py',
 '--request',linux+'/clock_request.json','--clearance',linux+'/clock_process/launch_clearance.json']
assert l['exact_wsl_arguments'] == args
for name,expected in [('run.ps1','cc57d8793bfd38fba3b1f5fc85a4bb46752801d0e862f82213beac265a82265a'),
 ('run_durable.ps1','ac35ca1f281559af9f9df2d345aceccdda4ef1091eb22bcb2c3eadb2faac3cdf')]:
    assert pins[(BASE/'clock_process'/name).as_posix()] == expected
absent=['clock_process/launch_clearance.json','clock_process/started.lock','clock_process/start.json',
        'clock_process/child.json','clock_process/exit.json','run','stage_receipts']
assert all(not (BASE/n).exists() for n in absent)
result=dict(passed=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 request_subject=dict(path=rp.as_posix(),sha256=sha(rp)),
 launch_receipt_subject=dict(path=lp.as_posix(),sha256=sha(lp)),
 checked_input_pins=len(pins),preserved_original_external_pins=external,all_current_input_pins_exact=True,
 unchanged_scope_fields=list(unchanged),exact_launch_arguments=True,original_roles_unchanged=True,
 source_and_saved_auditor_reviews_bound=True,dispatch_performed=False,execution_selected=False,
 root_selection_must_wait_for_no_active_policy_or_native_work=True,native_steps=0,model_calls=0,
 MJB_serializations=0,writer_sha256=sha(__file__),hardware_authorized=False)
with OUT.open('x',encoding='utf-8') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(path=OUT.as_posix(),sha256=sha(OUT),pins=len(pins),passed=True)))
