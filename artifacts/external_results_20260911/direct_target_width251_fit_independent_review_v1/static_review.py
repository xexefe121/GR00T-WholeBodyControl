"""Independent source-only comparisons; no numerical or task imports."""
import ast
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SOURCE=NEW/'direct_target_width251_student_v1/source_draft_v1'
OLD=NEW/'direct_target_causal_width512_student_v1/source_snapshot_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return ast.parse(p.read_text())
def member(p,name):return next(n for n in tree(p).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def named_calls(node,name):return [n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name]
def attr_calls(node,name):return [n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==name]

old_map=json.loads((NEW/'direct_target_causal_width512_student_v1/source_preparation.json').read_text())['source_sha256']
checks={}
checks['all_28_inherited_modules_exact']=len(old_map)==28 and all(sha(SOURCE/name)==digest==sha(OLD/name) for name,digest in old_map.items())
checks['promoted_class_exact']=same(member(SOURCE/'recovery_promoted.py','PromotedDirect'),member(OLD/'width512_promoted.py','PromotedDirect'))
new_export=member(SOURCE/'recovery_promoted.py','export_onnx')
old_export=member(OLD/'width512_promoted.py','export_onnx')
new_graph=next(n for n in ast.walk(new_export) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='make_graph')
old_graph=next(n for n in ast.walk(old_export) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='make_graph')
assert new_graph.args[1].value=='selected_recovery_width512_context_weights_fp64_execution'
new_graph.args[1]=old_graph.args[1]
checks['manual_export_only_graph_label_changed']=same(new_export,old_export)
for name in ('atomic','run_backend','target_errors','summarize','parity','save_drift'):
    checks['diagnostics_function_exact_'+name]=same(member(SOURCE/'recovery_diagnostics.py',name),member(OLD/'context_diagnostics.py',name))
main=member(SOURCE/'train_recovery.py','run_continuation')
checks['four_training_forward_sites']=len(named_calls(main,'counted_forward'))==4
checks['one_backward_site']=len(attr_calls(main,'backward'))==1
checks['one_optimizer_step_site']=len(attr_calls(main,'step'))==1
checks['one_warm_restore_site']=len(named_calls(main,'restore_warm512'))==1
checks['no_expansion_call']=not any(named_calls(main,k) for k in ('expand_source','expand_actor','verify_width_restoration','assert_blinded_columns'))
checks['initial_drift_old_corpora_only']='for corpus in OLD_CORPORA:' in ast.get_source_segment((SOURCE/'train_recovery.py').read_text(),member(SOURCE/'train_recovery.py','initial_drift'))
checks['unchanged_original_objective_imports']=all(sha(SOURCE/name)==sha(OLD/name) for name in ('direct_objective.py','full_state_objective.py','balanced_response_objective.py'))
result=dict(passed=all(checks.values()),source_only=True,checks=checks,
    source_sha256={p.name:sha(p) for p in sorted(SOURCE.glob('*.py'))},
    inherited_source_sha256=old_map,task_checkpoint_loads=0,task_array_loads=0,
    numerical_tests=0,model_forward_calls=0,gradient_calls=0,optimizer_steps=0,native_calls=0,
    protocol_updates_rates_coefficient_unselected=True,concrete_launch_clearance=False)
with (BASE/'static_review_v1.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(passed=result['passed'],checks=checks,source_files=len(result['source_sha256']))))
