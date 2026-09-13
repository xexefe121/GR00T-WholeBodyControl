"""Distinct provenance and producer labels; isolated AST and synthetic metrics only."""
from pathlib import Path
import ast
import pytest
from audit_balanced_math import ENERGY_PROVENANCE_RULE,PRODUCER_WEIGHT_RULE,energy_weights,balanced_metrics
from test_balanced_audit import energy_fixture,metric_fixture
BASE=Path(__file__).resolve().parent

def test_independent_literals_match_frozen_source_contract():
    producer=BASE.parent.parent/'direct_target_causal_response_balanced_student_v2/source_snapshot_v1/balance_contract.py'
    t=ast.parse(producer.read_text())
    actual=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign)
        and any(isinstance(k,ast.Name) and k.id=='WEIGHT_RULE' for k in n.targets))
    assert actual==PRODUCER_WEIGHT_RULE=='mean_six_teacher_group_energies_over_group_energy'
    assert ENERGY_PROVENANCE_RULE=='Emean/Eg' and PRODUCER_WEIGHT_RULE!=ENERGY_PROVENANCE_RULE

@pytest.mark.parametrize('role',['checkpoint','report'])
@pytest.mark.parametrize('rule,accepted',[(PRODUCER_WEIGHT_RULE,True),(ENERGY_PROVENANCE_RULE,False)])
def test_actual_auditor_metadata_checks_require_producer_label(role,rule,accepted):
    t=ast.parse((BASE/'audit_saved_warm.py').read_text())
    name=role+'_weight_rule'
    expr=next(n.args[1] for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='check' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value==name)
    assert eval(compile(ast.Expression(expr),'<isolated metadata check>','eval'),
        {role:{'response_weight_rule':rule},'PRODUCER_WEIGHT_RULE':PRODUCER_WEIGHT_RULE}) is accepted

def test_reconstructed_metrics_emit_producer_label():
    energy=energy_fixture();result=balanced_metrics(metric_fixture(energy),energy_weights(energy),energy)
    assert result['response_weight_rule']==PRODUCER_WEIGHT_RULE
    assert energy['rule']==ENERGY_PROVENANCE_RULE

def test_energy_document_keeps_its_own_provenance_label():
    energy=energy_fixture();energy['rule']=PRODUCER_WEIGHT_RULE
    with pytest.raises(AssertionError):energy_weights(energy)
