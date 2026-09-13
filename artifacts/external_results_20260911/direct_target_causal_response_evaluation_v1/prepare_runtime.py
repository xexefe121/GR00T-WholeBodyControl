"""Derive the unchanged native evaluator with single warm71000 release checks."""
import hashlib,json,shutil,difflib
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_context_evaluation_v2'
SOURCE=BASE/'source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
prep=read(OLD/'source_preparation.json')
assert sha(OLD/'source_preparation.json')=='fd8545cafdf8e5faf5fbd567dde9748c09bd0c581a2ea1a6e4c3de5c7e904359'
SOURCE.mkdir(exist_ok=False)
changed=('evaluation_gate.py','context_release.py','test_context_release.py')
for name,digest in prep['source_sha256'].items():
    source=OLD/'source_draft_v1'/name;assert sha(source)==digest,name
    if name in changed:continue
    target=SOURCE/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
gate=(OLD/'source_draft_v1/evaluation_gate.py').read_text()
for before,after in [
    ("binding['ordinary_final_step']==68000","binding['ordinary_final_step']==71000"),
    ("binding['controller']=='direct_absolute_target_1323_causal_context_study'","binding['controller']=='direct_absolute_target_1323_causal_response_balanced'"),
    ("binding['context_condition'] in ('blinded','causal')","binding['context_condition']=='causal'"),
    ("'paired_report','shared_manifest','context_alignment','blinded_fit_report','causal_fit_report'","'shared_manifest','context_alignment','energy_source'")]:
    assert gate.count(before)==1,before;gate=gate.replace(before,after)
(SOURCE/'evaluation_gate.py').write_text(gate)
shutil.copyfile(BASE/'context_release_new.py.txt',SOURCE/'context_release.py')
oldtest=(OLD/'source_draft_v1/test_context_release.py').read_text()
test=oldtest.replace('68000','71000').replace('optimizer_step=3000,fresh_optimizer=True','optimizer_step=6000,fresh_optimizer=False')
test=test.replace('from context_release import condition_report, counter, COEFFICIENT','from context_release import condition_report, counter, COEFFICIENT, GROUP_WEIGHTS, WEIGHT_RULE, SOURCE_CHECKPOINT')
test=test.replace("head_output='normalized_target',fixed_full_state_coefficient=COEFFICIENT,execution_dtype='float64',","head_output='normalized_target',fixed_full_state_coefficient=COEFFICIENT,execution_dtype='float64',\n        ordinary_start_step=68000,optimizer_start_step=3000,context_and_normalization_reused=True,\n        response_group_weights=list(GROUP_WEIGHTS),response_weight_rule=WEIGHT_RULE,source_checkpoint_sha256=SOURCE_CHECKPOINT,")
test=test.replace("@pytest.mark.parametrize('condition',['blinded','causal'])","@pytest.mark.parametrize('condition',['causal'])")
test=test.replace("('condition','blinded'),('ordinary_final_step',65000),('features',1000),","('condition','blinded'),('ordinary_final_step',68000),('features',1000),\n    ('ordinary_start_step',65000),('optimizer_start_step',0),('optimizer_step',3000),('fresh_optimizer',True),\n    ('context_and_normalization_reused',False),('response_group_weights',[1.]*6),\n    ('response_weight_rule','arbitrary'),('source_checkpoint_sha256','0'*64),")
(SOURCE/'test_context_release.py').write_text(test)
math=BASE.parent/'direct_target_response_balance_math_v1/source_prepared_v1/balance_contract.py'
assert sha(math)=='3bed156c9a15f2fe22eccd022330c745f7073481bcd36585e6c7b00abce76567'
shutil.copyfile(math,SOURCE/'balance_contract.py')
current={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob('*.py')}
unchanged={n:h for n,h in prep['source_sha256'].items() if n not in changed}
assert len(current)==37 and len(unchanged)==33
diff=''.join(''.join(difflib.unified_diff((OLD/'source_draft_v1'/n).read_text().splitlines(True),(SOURCE/n).read_text().splitlines(True),fromfile='old/'+n,tofile='new/'+n)) for n in changed)
(BASE/'runtime_changes.diff').write_text(diff)
write(BASE/'runtime_derivation.json',dict(preparation_only=True,original_preparation_sha256=sha(OLD/'source_preparation.json'),
 source_directory=SOURCE.as_posix(),source_sha256=current,unchanged_source_sha256=unchanged,changed_sources=list(changed),
 added_source='balance_contract.py',changes='Single causal warm71000 release gates only; native execution, features, commands and strict oracle unchanged.',
 actual_checkpoint_selected=False,task_model_calls=0,native_steps=0))
print(json.dumps(dict(source_files=len(current),unchanged=len(unchanged),derivation_sha256=sha(BASE/'runtime_derivation.json'))))
