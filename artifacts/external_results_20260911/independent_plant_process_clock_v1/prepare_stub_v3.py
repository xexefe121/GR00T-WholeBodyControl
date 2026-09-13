"""Preserve previous preparation; verify the narrow lifecycle fix with fake handles."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if __name__=='__main__':
    prior=json.loads((BASE/'source_preparation_v2.json').read_text())
    assert prior['source_preparation_passed']
    for p,s in prior['input_sha256'].items():assert sha(p)==s
    changed=[]
    for path in (BASE/'source_draft_v2').glob('*.py'):
        if sha(path)!=sha(BASE/'source_draft_v3'/path.name):changed.append(path.name)
    assert set(changed)=={'dummy_worker.py','test_scaffold.py'}
    pins=dict(prior['input_sha256'])
    for path in [BASE/'source_preparation_v2.json',Path(__file__),BASE/'V3_LIFECYCLE.md',*sorted((BASE/'source_draft_v3').glob('*.py'))]:pins[str(path)]=sha(path)
    with (BASE/'stub_request_v3.json').open('x') as f:f.write(json.dumps(dict(preparation_only=True,input_sha256=pins,changed_files=changed),indent=2)+'\n')
    with (BASE/'stub_stdout_v3.log').open('xb') as stdout,(BASE/'stub_stderr_v3.log').open('xb') as stderr:
        result=subprocess.run([sys.executable,'-B','-m','unittest','test_scaffold','-v'],cwd=str(BASE/'source_draft_v3'),
                              env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stdout,stderr=stderr)
    final=all(sha(p)==s for p,s in pins.items())
    report=dict(source_preparation_passed=result.returncode==0 and final,stub_exit_code=result.returncode,
        request_sha256=sha(BASE/'stub_request_v3.json'),input_sha256=pins,all_input_hashes_unchanged=final,
        changed_files=changed,stderr_sha256=sha(BASE/'stub_stderr_v3.log'),stdout_sha256=sha(BASE/'stub_stdout_v3.log'),
        actual_native_steps=0,actual_process_clock_runs=0,actual_worker_processes=0,model_calls=0,optimizer_updates=0,
        actual_run_selected=False,component_qualified=False)
    with (BASE/'source_preparation_v3.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['source_preparation_passed'],sha256=sha(BASE/'source_preparation_v3.json'),pins=len(pins))))
    sys.exit(result.returncode)
