import hashlib,json,sys,xml.etree.ElementTree as ET,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'direct_target_causal_response_balanced_student_v2'
OLD=NEW/'direct_target_causal_response_balanced_student_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep=read(BASE/'source_preparation.json');prior=read(OLD/'source_preparation.json')
assert sha(BASE/'source_preparation.json')=='403cea76732f52ccc0aa8e81da393e517a87e549c96e7be4ceb31a612550065a'
assert set(prep['source_sha256'])==set(prior['source_sha256']) and len(prep['source_sha256'])==25
changed=[]
for name,digest in prep['source_sha256'].items():
    actual=BASE/'source_prepared_v1'/name;old=OLD/'source_prepared_v1'/name
    assert sha(actual)==digest and sha(old)==prior['source_sha256'][name]
    if digest!=prior['source_sha256'][name]:changed.append(name)
assert changed==['train_response_balanced.py']
before=(OLD/'source_prepared_v1/train_response_balanced.py').read_text()
needle="write(dest/'request.json',dict(**request,condition=condition,initialization_sha256="
assert before.count(needle)==1
assert before.replace(needle,needle.replace('dict(**request','dict(request'))==(BASE/'source_prepared_v1/train_response_balanced.py').read_text()
for sub in prep['subjects'].values():assert sha(sub['path'])==sub['sha256']
failure=NEW/'direct_target_response_initial_failure_root_v1/report.json'
assert sha(failure)=='50c29d574ec3b17375a81b468da443ba6d80b759e257b51906ba054ed329e646'
assert read(failure)['saved_initialization_audit_pass'] is True
owner=read(OLD/'owner_completion_verification.json')
assert owner['accounting_passed'] is True and owner['completion_passed'] is False and owner['processes_absent'] is True
assert owner['raw_python_exit_code']==owner['exit_code']==1
run=subprocess.run([sys.executable,'-m','pytest','--rootdir=.','--confcutdir=.','-q','test_carried_request.py','--junitxml='+str(OUT/'root_regression_tests.xml')],cwd=BASE,capture_output=True,text=True)
(OUT/'root_regression_tests.log').write_text(run.stdout+'\n'+run.stderr)
assert run.returncode==0,run.stdout+run.stderr
result=dict(source_review_pass=True,data_export_source_review_pass=True,repair_review_pass=True,source_preparation_sha256=sha(BASE/'source_preparation.json'),
 source_sha256=prep['source_sha256'],changed_source=['train_response_balanced.py'],unchanged_sources=24,
 repair='dict(request, condition=...) safely overrides carried fields instead of duplicate expanded keyword arguments',
 root_regression_tests_passed=3,prior_failure_audit_sha256=sha(failure),prior_owner_sha256=sha(OLD/'owner_completion_verification.json'),
 original_training_data_math_optimizer_diagnostics_export_and_gates_unchanged=True,
 inherited_source_review_sha256=sha(NEW/'direct_target_response_root_source_review_v1/review.json'),
 task_checkpoint_loads=0,task_prediction_arrays_loaded=0,task_model_calls=0,gradient_calls=0,native_steps=0,
 actual_fit_selected=False,writer_sha256=sha(__file__))
path=OUT/'review.json'
with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(path),path=path.as_posix())))
