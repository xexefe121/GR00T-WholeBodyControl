"""Verify preserved originals and run deterministic source_v2 tests only."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if __name__=='__main__':
    prior=json.loads((BASE/'source_preparation.json').read_text())
    assert prior['source_preparation_passed']
    for p,s in prior['input_sha256'].items():assert sha(p)==s
    for name,s in prior['copied_sources'].items():assert sha(BASE/'source_draft_v2'/name)==s
    pins=dict(prior['input_sha256'])
    for p in [BASE/'source_preparation.json',Path(__file__),BASE/'V2_SCHEMA.md',*sorted((BASE/'source_draft_v2').glob('*.py'))]:pins[str(p)]=sha(p)
    request=dict(preparation_only=True,input_sha256=pins,proposed_controls=1819,proposed_steps=18190,
        actual_native_steps=0,actual_process_clock_runs=0,actual_worker_processes=0,model_calls=0,
        immutable_v1_preserved=True,original_component_files_unchanged=True)
    with (BASE/'stub_request_v2.json').open('x') as f:f.write(json.dumps(request,indent=2)+'\n')
    args=[sys.executable,'-B','-m','unittest','test_scaffold','-v']
    with (BASE/'stub_stdout_v2.log').open('xb') as stdout,(BASE/'stub_stderr_v2.log').open('xb') as stderr:
        result=subprocess.run(args,cwd=str(BASE/'source_draft_v2'),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stdout,stderr=stderr)
    exact=all(sha(p)==s for p,s in pins.items())
    report=dict(source_preparation_passed=result.returncode==0 and exact,stub_exit_code=result.returncode,
        request_sha256=sha(BASE/'stub_request_v2.json'),input_sha256=pins,all_inputs_unchanged=exact,
        stderr_sha256=sha(BASE/'stub_stderr_v2.log'),stdout_sha256=sha(BASE/'stub_stdout_v2.log'),
        actual_native_steps=0,actual_process_clock_runs=0,actual_worker_processes=0,model_calls=0,
        optimizer_updates=0,source_v1_preserved=True,component_qualified=False,actual_run_selected=False)
    with (BASE/'source_preparation_v2.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['source_preparation_passed'],source_preparation_sha256=sha(BASE/'source_preparation_v2.json'),pins=len(pins))))
    sys.exit(result.returncode)
