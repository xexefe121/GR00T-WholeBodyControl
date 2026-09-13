"""Compare literal frozen-source schemas through isolated AST, never import fitter."""
import ast
from pathlib import Path
from audit_width_math import restoration_fields
from audit_context_math import WEIGHT

BASE=Path(__file__).resolve().parent
PRODUCER=BASE.parent.parent/'direct_target_causal_width512_student_v1/source_prepared_v1'
def tree(path):return ast.parse(path.read_text())
def evaluate(value,env):return eval(compile(ast.Expression(value),'<metadata only>','eval'),env)
def assigned(t,name):
    return next(n.value for n in ast.walk(t) if isinstance(n,ast.Assign)
        and any(isinstance(a,ast.Name) and a.id==name for a in n.targets))

def test_request_expected_literals_match_frozen_protocol():
    producer=tree(PRODUCER/'response_contract.py');audit=tree(BASE/'audit_saved_warm.py')
    expected=evaluate(assigned(producer,'expected'),dict(UPDATES=10000,START_STEP=71000,FINAL_STEP=81000,
        OPTIMIZER_START=6000,OPTIMIZER_FINAL=16000,COEFFICIENT=WEIGHT,
        BUDGETS=evaluate(assigned(audit,'expected_budget'),{})))
    loop=next(n for n in ast.walk(audit) if isinstance(n,ast.For)
        and any(isinstance(v,ast.Constant) and v.value=='request.' for v in ast.walk(n)))
    ours=evaluate(loop.iter.func.value,dict(WEIGHT=WEIGHT,expected_budget=expected['budgets']))
    assert ours==dict(expected,root_selected=True)

def test_restoration_literal_schema_matches_actual_saver():
    t=tree(PRODUCER/'width_restore.py')
    returned=next(n.value for n in ast.walk(t) if isinstance(n,ast.Return))
    assert evaluate(returned,dict(SEED=20260912))==restoration_fields()

def test_report_and_checkpoint_metadata_keys_exist_in_frozen_writer():
    producer=tree(PRODUCER/'train_response_balanced.py');audit=tree(BASE/'audit_saved_warm.py')
    keys={name:{k.arg for k in assigned(producer,name).keywords} for name in ('checkpoint','report')}
    required_report=set();required_checkpoint=set()
    for n in ast.walk(audit):
        if not isinstance(n,ast.For) or not isinstance(n.iter,ast.Call) or not isinstance(n.iter.func,ast.Attribute):continue
        if not isinstance(n.iter.func.value,ast.Call) or not isinstance(n.iter.func.value.func,ast.Name) or n.iter.func.value.func.id!='dict':continue
        strings=[v.value for v in ast.walk(n) if isinstance(v,ast.Constant) and isinstance(v.value,str)]
        if '_report.' in strings or 'report_extra.' in strings:required_report.update(k.arg for k in n.iter.func.value.keywords)
        if '_checkpoint.' in strings:required_checkpoint.update(k.arg for k in n.iter.func.value.keywords)
    assert len(required_report)==27 and len(required_checkpoint)==13
    assert required_report<=keys['report'] and required_checkpoint<=keys['checkpoint']

def test_actual_initial_and_report_drift_metadata_no_extra_byte_gate():
    t=tree(PRODUCER/'train_response_balanced.py');function=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='initial_drift')
    returned=next(n.value for n in function.body if isinstance(n,ast.Return))
    result=evaluate(returned,dict(maximum=0.,result={}))
    assert result['passed'] is True and result['byte_gate_required'] is False
    assert result['stored_hidden_width']==512 and result['stored_first_layer_width']==1323
    assert result['first_layer_execution']=='split_old256_new256_original1000_plus323'
    assert result['source_ordinary_step']==71000 and result['changed_MatMul_dimension'] is True
