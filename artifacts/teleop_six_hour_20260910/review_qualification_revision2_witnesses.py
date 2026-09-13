"""Execute actual revision2 validation expressions on malformed-evidence witnesses."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/teleop_six_hour_20260910/qualify_recorded_candidate.py'
OUT=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/canonical_qualification_independent_review_v1')
blob=SOURCE.read_bytes();tree=ast.parse(blob)
warnings_expr=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='warnings_clear' for t in n.targets))
shape_expr=next(n.test for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(s,ast.Raise) and isinstance(s.exc,ast.Call)
    and any(isinstance(a,ast.Constant) and a.value=='physical trace dimensions differ' for a in s.exc.args) for s in n.body))
rows=[]
for name,value,expected in [('eight_integer_zeros',[0]*8,True),('nonzero_warning',[0,1,0,0,0,0,0,0],False),
                           ('missing_evidence',None,False),('wrong_count',[0]*7,False),('float_zero_values',[0.]*8,False),('boolean_false_values',[False]*8,False)]:
    result=bool(eval(compile(ast.Expression(warnings_expr),'<auditor-warnings-expression>','eval'),{'warnings':value}))
    assert result==expected
    rows.append(dict(kind='warning_gate',name=name,accepted=result,expected=expected,passed=True))
for name,shape,expected_reject in [('23_actuators',(10,23),False),('one_broadcast_column',(10,1),True),('flat_torque_vector',(10,),True),('wrong_actuator_count',(10,22),True)]:
    arrays={'physics_torque':np.zeros(shape),'physics_qpos':np.zeros((11,30)),'physics_qvel':np.zeros((11,29))}
    rejected=bool(eval(compile(ast.Expression(shape_expr),'<auditor-physics-shape-expression>','eval'),{'arrays':arrays,'nphysics':10}))
    assert rejected==expected_reject
    rows.append(dict(kind='torque_shape_gate',name=name,rejected=rejected,expected_reject=expected_reject,passed=True))
assert SOURCE.read_bytes()==blob
result=dict(kind='independent_revision2_malformed_evidence_witnesses',auditor_revision=2,
    auditor_sha256=hashlib.sha256(blob).hexdigest(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    expressions_extracted_from_actual_source=True,witnesses=rows,all_pass=True,no_dynamics_rerun=True,
    implication='Revision2 resolves the two reported evidence-validation gaps. Four actual A/B runs already passed these independent checks, so their failure outcomes remain unchanged.',
    remaining_label_nuance='source_controls counts a terminal partial slot; full-source/lifecycle gates still reject incomplete slots.')
path=OUT/'auditor_revision2_review.json';assert not path.exists()
path.write_text(json.dumps(result,indent=2))
(OUT/'qualify_recorded_candidate_revision2_snapshot.py').write_bytes(blob)
(OUT/'revision2_witness_producer_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(all_pass=True,witnesses=len(rows),auditor_sha256=result['auditor_sha256'])))
