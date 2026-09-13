"""Independent helper source/metadata tests. No runtime inventory regeneration or task calls."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
TARGET=NEW/'direct_target_causal_width512_evaluation_v1'
PRIOR=NEW/'direct_target_causal_response_evaluation_v1'
PREP=TARGET/'launch_helper_preparation_v2.json'
EXPECTED='c82159677d483214d9a5e4b73a1173f2da5e8843afd7f51109dd753260ae2c22'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as stream:json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n')
def main():
    assert sha(PREP)==EXPECTED
    prep=read(PREP);mapping=prep['helper_sha256'];assert len(mapping)==8 and prep['source_preparation_passed'] is True
    pins={PREP.as_posix():EXPECTED};identical=[];changed=[]
    for name,digest in mapping.items():
        path=TARGET/name;old=PRIOR/name
        assert sha(path)==digest and sha(old)==prep['original_helper_sha256'][name]
        ast.parse(path.read_text(encoding='utf-8-sig'))
        pins[path.as_posix()]=digest;pins[old.as_posix()]=sha(old)
        (identical if path.read_bytes()==old.read_bytes() else changed).append(name)
    assert set(identical)=={'prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py'}
    for item in prep['subjects'].values():
        assert sha(item['path'])==item['sha256'];pins[item['path']]=item['sha256']
    runtime=read(TARGET/'source_preparation.json')
    assert runtime['source_sha256']==prep['source_sha256']
    for name,digest in runtime['source_sha256'].items():
        path=Path(runtime['source_directory'])/name;assert sha(path)==digest;pins[path.as_posix()]=digest
    inventory_path=TARGET/'runtime_inventory.json'
    assert sha(inventory_path)=='6a6f771154e3cbbc7660713f77a12df6592fcf39f0962c34739c58101cfc7f1e'
    pins[inventory_path.as_posix()]=sha(inventory_path);inventory=read(inventory_path)
    assert inventory['source_sha256']==runtime['source_sha256']
    assert inventory['recursive_training_hashes'] is False and inventory['baseline_full291_fixture_byteexact'] is True
    entries=inventory['files'];paths=[Path(e['path']).resolve().as_posix().casefold() for e in entries]
    assert len(entries)==inventory['total_files']==5194 and len(set(paths))==5194
    assert sum(e['bytes'] for e in entries)==inventory['total_bytes']
    forbidden=('/direct_target_gpu_20260911/','/one_step_policy_branch_collection_resume2969_v1/collection/data/',
        '/direct_target_full_state_secants_v1/generation/')
    assert not any(any(x in p for x in forbidden) for p in paths)
    source_review=NEW/'direct_target_width512_evaluation_independent_review_v2/review.json'
    assert read(source_review)['source_sha256']==runtime['source_sha256'] and read(source_review)['passed'] is True
    pins[source_review.as_posix()]=sha(source_review)
    for name in ('source_preparation.json','launch_helper_changes.diff'):
        pins[(TARGET/name).as_posix()]=sha(TARGET/name)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):env[key]='1'
    env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(TARGET)
    args=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','test_launch_helpers.py','test_release_helpers.py','--junitxml='+str(BASE/'synthetic_tests.xml')]
    with (BASE/'synthetic_tests.log').open('xb') as stream:
        run=subprocess.run(args,cwd=TARGET,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
    suites=list(ET.parse(BASE/'synthetic_tests.xml').getroot().iter('testsuite'))
    counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    write(BASE/'test_exit.json',dict(exit_code=run.returncode,arguments=args,counts=counts))
    assert run.returncode==0 and counts==dict(tests=41,failures=0,errors=0,skipped=0),counts
    for path,digest in pins.items():assert sha(path)==digest,path
    result=dict(passed=True,helper_review_pass=True,preparation_only=True,
        helper_preparation_subject=dict(path=PREP.as_posix(),sha256=EXPECTED),helper_sha256=mapping,
        source_sha256=runtime['source_sha256'],input_sha256=pins,byte_identical_helpers=identical,changed_helpers=changed,
        independent_synthetic_tests=counts,tests_sha256=sha(BASE/'synthetic_tests.xml'),
        test_exit_sha256=sha(BASE/'test_exit.json'),review_source_sha256=sha(__file__),findings=[],
        inventory_metadata_checked=True,inventory_files=5194,inventory_payloads_rehashed_in_this_review=False,
        full_source_and_helper_hashes_unchanged=True,
        contracts=['Actual fit/audit/owner/release/source/helper positives and16 direct subjects required before binding.',
            'Explicit causal81000 width512 namespace and1323 input; original1569+conditional250 and one separate witness.',
            'CreateNew launch guards, hidden held child handle, known raw exit, separate diagnostic result and failed-prefix preservation.',
            'Final review names exact binding/receipt path and SHA; pre/post frozen inputs and saved process identity checked.',
            'Four unchanged helpers byte-exact; remaining changes are endpoint/namespace/shape/documentation and corresponding test expectation.'],
        task_arrays_loaded=0,task_checkpoint_loads=0,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,
        actual_binding_created=False,actual_launches=0,
        limitations=['Runtime inventory payloads remain subject to concrete launch rehash; this review checks declared inventory metadata and exact code.',
            'No final model or behavioral qualification; actual fit/audit/owner and per-stage launch receipts remain required.'])
    write(BASE/'review.json',result)
    print(json.dumps(dict(passed=True,tests=41,review_sha256=sha(BASE/'review.json'))))
if __name__=='__main__':main()
