"""Read-only source comparison and unbound-gate check; no model construction."""
import ast,importlib.util,json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent;DRAFT=BASE/'source_draft_v1';OLD=BASE.parent/'velocity_chord_student_evaluation_v1/source_snapshot_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
unchanged=[];updated=[]
for original in OLD.rglob('*.py'):
    relative=original.relative_to(OLD);derived=DRAFT/relative
    if str(relative) in ('evaluation_gate.py','head_activation_witness.py'):
        expected=original.read_text().replace('70000','75000').replace('evaluate_velocity_chord_student.py','evaluate_physical_response_student.py').replace('one_fixed_velocity_chord_fit','one_fixed_physical_response_fit')
        assert derived.read_text()==expected;updated.append(str(relative))
    else:assert sha(derived)==sha(original),relative;unchanged.append(str(relative))
old=(OLD/'evaluate_velocity_chord_student.py').read_text()
expected=old.replace('70000','75000').replace('evaluate_velocity_chord_student.py','evaluate_physical_response_student.py').replace('one_fixed_velocity_chord_fit','one_fixed_physical_response_fit')
actual=(DRAFT/'evaluate_physical_response_student.py').read_text();assert actual==expected
class Normalize(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value==75000:node.value=70000
        elif isinstance(node.value,str):node.value=node.value.replace('75000','70000').replace('one_fixed_physical_response_fit','one_fixed_velocity_chord_fit')
        return node
    def visit_keyword(self,node):
        if node.arg=='one_fixed_physical_response_fit':node.arg='one_fixed_velocity_chord_fit'
        return self.generic_visit(node)
assert ast.dump(ast.parse(old),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(actual)),include_attributes=False)
spec=importlib.util.spec_from_file_location('future_eval_gate_readonly',DRAFT/'evaluation_gate.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
try:module.require_ready(BASE)
except ValueError as e:assert 'unbound' in str(e);unbound=True
else:raise AssertionError('Unbound preparation unexpectedly passed evaluation gate')
assert not (BASE/'evaluation_binding.json').exists() and not (BASE/'witness_binding.json').exists()
result=dict(passed=True,unchanged_original_sources=unchanged,updated_subject_gates=updated,new_evaluator_entire_AST_equal_after_subject_normalization=True,
    schedule_and_runtime_unchanged=True,unbound_gate_rejected_before_model_creation=unbound,real_graph_calls=0,physics_steps=0,
    source_review_ready=True,execution_or_witness_authorized=False,actual_final75000_artifacts_bound=False,
    source_sha256={p.relative_to(DRAFT).as_posix():sha(p) for p in DRAFT.rglob('*.py')},
    freeze_requirement='Copy only declared .py sources into a new source_snapshot; do not copy draft __pycache__. Bind actual final artifacts/reviews only after a selected completed fit.')
assert not (BASE/'source_only_check.json').exists();(BASE/'source_only_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','unchanged_original_sources')}))
