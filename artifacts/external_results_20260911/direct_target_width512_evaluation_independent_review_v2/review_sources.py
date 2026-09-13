"""Independent source/hash review and synthetic CPU tests; no task models or arrays."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
TARGET=NEW/'direct_target_causal_width512_evaluation_v1'
PREP=TARGET/'source_preparation.json'
EXPECTED='5703e68e439963c271dd79d5d0f30aaa3be9683f2d28096502aedb3747b96962'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as stream:json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n')

def main():
    assert sha(PREP)==EXPECTED
    prep=read(PREP);source=Path(prep['source_directory'])
    assert prep['source_preparation_passed'] is True and len(prep['source_sha256'])==38
    old_prep_path=NEW/'direct_target_causal_response_evaluation_v1/source_preparation.json'
    old=read(old_prep_path);old_source=Path(old['source_directory'])
    pins={PREP.as_posix():EXPECTED,old_prep_path.as_posix():sha(old_prep_path),**prep['input_sha256']}
    prior=NEW/'direct_target_width512_evaluation_independent_review_v1'
    for name in ('review_sources.py','test_exit.json','synthetic_tests.xml','synthetic_tests.log','prepare_scipy_review.py'):
        pins[(prior/name).as_posix()]=sha(prior/name)
    same=[];changed=[];added=[]
    for name,digest in prep['source_sha256'].items():
        path=source/name
        assert sha(path)==digest,name
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        pins[path.as_posix()]=digest
        if name in old['source_sha256']:
            original=old_source/name
            assert sha(original)==old['source_sha256'][name]
            pins[original.as_posix()]=sha(original)
            (same if path.read_bytes()==original.read_bytes() else changed).append(name)
        else:added.append(name)
    assert len(same)==34 and sorted(changed)==['context_release.py','evaluation_gate.py','test_context_release.py']
    assert added==['test_width_release.py']
    assert sha(source/'evaluate_direct_target_student.py')=='061e2a3145c6ed273bf7ac915efec0f1866ba712ccb8ba571e8c08417243d970'
    for path,digest in pins.items():assert sha(path)==digest,path
    write(BASE/'pretest_pins.json',dict(passed=True,input_sha256=pins,source_sha256=prep['source_sha256']))
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        env[name]='1'
    env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(source)
    files=['test_causal_features.py','test_context_release.py','test_direct_adapter.py','test_release_gate.py','test_width_release.py']
    arguments=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*files,'--junitxml='+str(BASE/'synthetic_tests.xml')]
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (BASE/'synthetic_tests.log').open('xb') as stream:
        result=subprocess.run(arguments,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
    suites=list(ET.parse(BASE/'synthetic_tests.xml').getroot().iter('testsuite'))
    counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    write(BASE/'test_exit.json',dict(exit_code=result.returncode,started_utc=started,arguments=arguments,
        counts=counts,thread_limits={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}))
    assert result.returncode==0 and counts==dict(tests=121,failures=0,errors=0,skipped=0),counts
    for path,digest in pins.items():assert sha(path)==digest,path
    review=dict(passed=True,source_review_pass=True,preparation_only=True,
        source_preparation_subject=dict(path=PREP.as_posix(),sha256=EXPECTED),source_sha256=prep['source_sha256'],
        input_sha256=pins,unchanged_source_files=same,changed_source_files=changed,added_source_files=added,
        independent_synthetic_tests=counts,tests_sha256=sha(BASE/'synthetic_tests.xml'),
        log_sha256=sha(BASE/'synthetic_tests.log'),test_exit_sha256=sha(BASE/'test_exit.json'),
        review_source_sha256=sha(__file__),all_pre_post_source_and_input_pins_exact=True,
        findings=[],prior_test_collection_failure='Isolated training Python lacks SciPy; unchanged suite rerun in existing SciPy CPU Python. No source modification or package installation.',reviewed_contracts=[
            '81000 ordinary checkpoint;512 width;10000 updates;6000-to16000 warm moments;seed20260912.',
            'Actual selected request ramp250 thencosine9750, context1323, fixed coefficient and group weights match frozen fit source.',
            '16 actual release roles bind fit/report/PT/head/norm/request/frozen/output/data/energy plus independent audit and owner.',
            'Original initial1e-5 gate remains through qualified trainer/owner/audit; no byte-only threshold added.',
            'Original CPU64/GPU64/ORT64 final1e-5 admission; finalGPU32 drift remains diagnostic.',
            '1569 controls plus conditional250 hold, one separate WSL witness, all strict physics/initial/history/target semantics unchanged.',
            'Every runtime .py source remains bound before model creation; absent or failed actual receipts reject activation.'
        ],task_checkpoint_loads=0,task_arrays_loaded=0,task_model_calls=0,ORT_calls=0,BFM_calls=0,native_steps=0,
        optimizer_updates=0,actual_binding_created=False,actual_execution_selected=False,
        limitations=['Source/synthetic qualification only; final fit/export/owner/audit and concrete per-stage launch reviews remain required.'])
    write(BASE/'review.json',review)
    print(json.dumps(dict(passed=True,tests=121,source_count=38,review_sha256=sha(BASE/'review.json'))))

if __name__=='__main__':main()
