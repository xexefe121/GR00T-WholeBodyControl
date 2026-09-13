"""Freeze source-only preparation and execute deterministic fake tests once."""
from pathlib import Path
import difflib,hashlib,json,re,subprocess,sys
BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_draft_v1';ORIGINAL=BASE/'source_original_v1';NEW=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(data,stream,indent=2);stream.write('\n')
def main():
    assert not (BASE/'source_preparation.json').exists()
    initial={p.name:sha(p) for p in sorted(SOURCE.glob('*.py'))}
    original={p.name:sha(p) for p in sorted(ORIGINAL.glob('*.py'))}
    assert len(initial)==32 and len(original)==24
    assert all(sha(NEW/'independent_plant_pending_result_v1/source_draft_v1'/name)==value for name,value in original.items())
    sys.path.insert(0,str(SOURCE))
    from check_integration_derivation import check
    derivation=check(ORIGINAL,SOURCE)
    write(BASE/'derivation_checks.json',derivation)
    changed=[name for name,value in original.items() if initial[name]!=value]
    diff=''
    for name in changed:
        diff+=''.join(difflib.unified_diff((ORIGINAL/name).read_text().splitlines(True),
            (SOURCE/name).read_text().splitlines(True),fromfile='original/'+name,tofile='integrated/'+name))
    for name in initial.keys()-original.keys():
        diff+=''.join(difflib.unified_diff([], (SOURCE/name).read_text().splitlines(True),fromfile='/dev/null',tofile='integrated/'+name))
    with (BASE/'integration_final.diff').open('x',encoding='utf-8') as f:f.write(diff)
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-v'],cwd=SOURCE,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    with (BASE/'integrated_fake_tests.log').open('x',encoding='utf-8') as f:f.write(result.stdout)
    assert result.returncode==0,result.stdout[-5000:]
    match=re.search(r'Ran (\d+) tests',result.stdout);assert match and int(match[1])==128
    assert all(sha(SOURCE/name)==value for name,value in initial.items())
    antecedents=[NEW/'independent_plant_pending_result_v1/source_preparation.json',
        NEW/'independent_pending_result_source_review_v1/review.json',
        NEW/'independent_plant_timing_instrumentation_v1/source_preparation_v2.json',
        NEW/'independent_timing_probe_root_review_v1/review.json']
    for path in antecedents:assert path.is_file(),str(path)
    evidence={p.name:sha(p) for p in sorted(BASE.iterdir()) if p.is_file() and p.name!='source_preparation.json'}
    report=dict(passed=True,source_preparation_only=True,source_sha256=initial,
        original_source_sha256=original,changed_original_modules=changed,
        unchanged_original_modules=[name for name in original if name not in changed],
        evidence_sha256=evidence,antecedent_sha256={str(p):sha(p) for p in antecedents},
        tests=128,original_tests=67,probe_tests=31,new_integration_tests=30,
        failures=0,errors=0,skips=0,test_python=sys.version,test_executable=sys.executable,
        preservation_directory=str(ORIGINAL),actual_run_selected=False,actual_request_created=False,
        actual_clock_calls=0,actual_gc_callbacks_installed=0,worker_processes=0,native_steps=0,model_calls=0,
        main_controls=1569,hold_controls=250,native_step_budget=18190,serialization_budget=4,
        actual_runtime_and_sidecar_audit_review_pending=True)
    write(BASE/'source_preparation.json',report)
    print(json.dumps({'passed':True,'tests':128,'preparation_sha256':sha(BASE/'source_preparation.json'),
        'changed_source':{name:initial[name] for name in changed},'new_source':{name:initial[name] for name in initial.keys()-original.keys()}},indent=2))
if __name__=='__main__':main()
