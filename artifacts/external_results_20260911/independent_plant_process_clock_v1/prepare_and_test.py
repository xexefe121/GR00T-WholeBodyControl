"""Copy reviewed components and run only deterministic stub tests once."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
NATIVE=NEW/'independent_native_stepper_v1/source_draft_v3'
TRANSPORT=NEW/'independent_plant_mailbox_transport_v1/source_snapshot_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__=='__main__':
    sources=['bfm_observations.py','capture_schema.py','clock_core.py','history.py','mailbox.py','model_identity.py','native_stepper.py','oracle_source.py']
    pins={};copies={}
    for name in sources:
        original=NATIVE/name;dest=BASE/'source_draft'/name
        with dest.open('xb') as f:f.write(original.read_bytes())
        assert sha(dest)==sha(original)
        pins[str(original)]=sha(original);copies[name]=sha(dest)
    original=TRANSPORT/'shared_mailbox.py';dest=BASE/'source_draft/shared_mailbox.py'
    with dest.open('xb') as f:f.write(original.read_bytes())
    pins[str(original)]=sha(original);copies[dest.name]=sha(dest)
    for relative in ['independent_native_stepper_root_review_v2/review_v3.json',
        'independent_plant_mailbox_transport_root_review_v1/review.json',
        'independent_plant_mailbox_transport_wsl_v1/owner_completion.json',
        'independent_native_stepper_saved_audit_v1/results_v1/report.json']:
        pins[str(NEW/relative)]=sha(NEW/relative)
    for path in [Path(__file__),BASE/'DESIGN.md',*sorted((BASE/'source_draft').glob('*.py'))]:pins[str(path)]=sha(path)
    request=dict(kind='process_clock_component_source_and_stub_only',input_sha256=pins,copied_sources=copies,
        proposed_actual_controls=1819,proposed_native_steps=18190,actual_native_steps=0,actual_process_runs=0,
        model_calls=0,optimizer_updates=0,scope='Deterministic fake clocks, fake steppers, in-process byte slots and fake process handles only.')
    with (BASE/'stub_request.json').open('x') as f:f.write(json.dumps(request,indent=2)+'\n')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    args=[sys.executable,'-B','-m','unittest','test_scaffold','-v']
    with (BASE/'stub_stdout.log').open('xb') as stdout,(BASE/'stub_stderr.log').open('xb') as stderr:
        result=subprocess.run(args,cwd=str(BASE/'source_draft'),env=env,stdout=stdout,stderr=stderr)
    final={p:sha(p) for p in pins}
    receipt=dict(source_preparation_passed=result.returncode==0 and final==pins,stub_exit_code=result.returncode,
        request_sha256=sha(BASE/'stub_request.json'),input_sha256=pins,all_original_and_final_pins_exact=final==pins,
        copied_sources=copies,stdout_sha256=sha(BASE/'stub_stdout.log'),stderr_sha256=sha(BASE/'stub_stderr.log'),
        actual_native_steps=0,actual_process_clock_runs=0,actual_spawned_workers=0,actual_models=0,optimizer_updates=0,
        stub_runner_subprocess=True,component_qualified=False,actual_run_selected=False)
    with (BASE/'source_preparation.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(passed=receipt['source_preparation_passed'],source_preparation_sha256=sha(BASE/'source_preparation.json'),pins=len(pins))))
    sys.exit(result.returncode)
