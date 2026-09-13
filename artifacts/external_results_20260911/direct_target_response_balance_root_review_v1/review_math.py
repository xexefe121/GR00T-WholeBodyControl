import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pytest
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'direct_target_response_balance_math_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep_path=BASE/'source_preparation.json'
assert sha(prep_path)=='caa20eb137970dff50b49880a6bfd4bef4deb15eaa24d7c5517a5fd21bbf89c2'
prep=read(prep_path);src=Path(prep['source_directory'])
assert prep['source_preparation_passed'] is True and prep['synthetic_CPU_tests_passed']==16
for n,h in prep['source_sha256'].items():assert sha(src/n)==h
for p in prep['subjects'].values():assert sha(p['path'])==p['sha256']
energy=read(BASE/'energy_source.json')
for p,h in energy['input_sha256'].items():assert sha(p)==h
for name,h in prep['original_modules_byte_exact'].items():
    assert sha(src/name)==h==sha(NEW/'direct_target_causal_context_study_v2/source_snapshot_v1'/name)
sys.path.insert(0,str(src))
from balance_contract import ZERO_RESPONSE_ENERGIES,GROUP_WEIGHTS,MEAN_ZERO_RESPONSE_ENERGY,CELL_GROUPS
e=np.asarray(ZERO_RESPONSE_ENERGIES,np.float64)
assert e.mean()==MEAN_ZERO_RESPONSE_ENERGY
assert tuple(e.mean()/e)==GROUP_WEIGHTS and CELL_GROUPS==tuple(range(6))*9
assert energy['zero_response_energies']==list(ZERO_RESPONSE_ENERGIES)
assert energy['group_weights']==list(GROUP_WEIGHTS)
for condition in ('causal','blinded'):
    metrics=read(NEW/'direct_target_causal_context_study_v2/fit'/condition/'ORT64_metrics.json')
    assert [c['zero_response_MSE'] for c in metrics['full_state_group_comparison']]==list(ZERO_RESPONSE_ENERGIES)
    cells=metrics['full_state_cells']
    assert len(cells)==54 and [c['tangent_group'] for c in cells]==list(CELL_GROUPS)
    actual=np.asarray([c['zero_response_MSE'] for c in cells]).reshape(9,6).mean(0)
    np.testing.assert_allclose(actual,e,rtol=0,atol=1e-18)
code=pytest.main([str(src/'test_balance_math.py'),'-q','--rootdir='+str(src),'--confcutdir='+str(src),'--junitxml='+str(OUT/'root_tests.xml')])
assert code==0
for n,h in prep['source_sha256'].items():assert sha(src/n)==h
r=dict(passed=True,math_review_pass=True,source_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_subject=dict(path=prep_path.as_posix(),sha256=sha(prep_path)),source_sha256=prep['source_sha256'],
 energy_source_sha256=sha(BASE/'energy_source.json'),original_modules_byte_exact=prep['original_modules_byte_exact'],
 original54cells_unchanged=True,group_order_verified=True,teacher_only_weights=list(GROUP_WEIGHTS),synthetic_CPU_tests_passed=16,
 test_xml_sha256=sha(OUT/'root_tests.xml'),writer_sha256=sha(__file__),
 task_prediction_arrays_read=0,task_checkpoint_reads=0,task_model_calls=0,task_gradient_calls=0,native_steps=0,
 fit_execution_selected=False,optimizer_or_learning_rate_selected_by_this_review=False,
 limitation='One balanced engineering continuation cannot isolate improvement from more epochs or optimizer choices; physical acceptance unchanged.')
p=OUT/'review.json'
with p.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(path=p.as_posix(),sha256=sha(p),tests=16,passed=True)))
