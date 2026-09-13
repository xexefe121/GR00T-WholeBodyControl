"""Source-only future evaluation preparation; no bindings, model or execution."""
import ast,json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent;DRAFT=BASE/'source_draft_v1'
OLD=BASE.parent/'velocity_chord_student_evaluation_v1/source_snapshot_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
substitutions=[]
for name in ('evaluation_gate.py','head_activation_witness.py','evaluate_velocity_chord_student.py'):
    text=(OLD/name).read_text();changed=text.replace('70000','75000').replace('evaluate_velocity_chord_student.py','evaluate_physical_response_student.py').replace('one_fixed_velocity_chord_fit','one_fixed_physical_response_fit')
    dest='evaluate_physical_response_student.py' if name=='evaluate_velocity_chord_student.py' else name
    (DRAFT/dest).write_text(changed);ast.parse(changed)
    substitutions.append(dict(original=name,derived=dest,original_sha256=sha(OLD/name),derived_sha256=sha(DRAFT/dest),changes='Only ordinary-final75000 identity, source filename and experiment reporting flag. Controller/math unchanged.'))
tree1=ast.parse((OLD/'evaluate_velocity_chord_student.py').read_text());tree2=ast.parse((DRAFT/'evaluate_physical_response_student.py').read_text())
for name in ('get_state','assess','new_trace','source_metrics','zero_parity'):
    a=next(n for n in tree1.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in tree2.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
assert sha(DRAFT/'student_linear_runtime.py')==sha(OLD/'student_linear_runtime.py')
assert sha(DRAFT/'proposal_evidence.py')==sha(OLD/'proposal_evidence.py')
assert not (BASE/'evaluation_binding.json').exists() and not (BASE/'witness_binding.json').exists()
receipt=dict(preparation_only=True,fit_selected=False,witness_selected=False,canonical_execution_selected=False,head_artifact_bound=False,
    source_substitutions=substitutions,strict_oracle_and_metrics_AST_unchanged=True,raw_history_runtime_and_proposal_evidence_byteexact=True,
    inputs_expected_later='Actual ordinary75000 fit/export, root data+fit audits, source/export clearances, separately authorized exactWSLbatch1 firstactivation witness.',
    graph_calls=0,physics_steps=0,optimizer_updates=0,source_sha256={p.relative_to(DRAFT).as_posix():sha(p) for p in DRAFT.rglob('*.py')})
(BASE/'source_preparation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_sha256','source_substitutions')}))
