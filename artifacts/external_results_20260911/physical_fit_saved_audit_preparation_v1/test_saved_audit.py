"""Synthetic saved-array and source tests only; no model/optimizer/ORT/native."""
import ast
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
SOURCE=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
sys.path.insert(0,str(SOURCE))
from audit_physical_fit_math import Proof,identities,phase_cells,budgets,physical_metrics,logged_cell_reductions,compose_three

def synthetic():
    centers=dict(base_target=np.zeros((3,23)),expert_target=np.zeros((3,23)),
        joint_limits=np.tile([-10.,10.],(23,1)))
    cells=[np.array([0,1],np.int64)]+[np.empty(0,np.int64) for _ in range(8)]
    physical=dict(base=np.zeros((2,23)),target=np.zeros((2,23)),successor=np.array([1,2]),
        successor_control=np.array([252,290]),cells=cells)
    physical['coverage']=[dict(dataset=i//3,phase=('acquisition','source','return')[i%3],
        requested=(99,819,100)[i%3],valid=len(ids),strict_failed=(99,819,100)[i%3]-len(ids),
        effective_mass=len(ids)/(9*(99,819,100)[i%3]),empty=len(ids)==0) for i,ids in enumerate(cells)]
    for key in ('policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped','successor_replan_boundary','successor_zero_gain'):
        physical[key]=np.array([True,False])
    return centers,physical

def test_all_requested_clock_identities():
    d,c,i=identities().T
    assert len(d)==3054 and np.array_equal(c,np.tile(np.arange(250,1268),3))
    assert np.array_equal(i,d*1019+c-250)
    cells=phase_cells(d,c+1,True)
    assert list(map(len,cells))==[99,819,100]*3
    assert len(np.unique(i+1))==3054 and np.all((i+1)%1019!=0)
    for code in range(3):
        assert np.all(d[i//1019==code]==code)
        assert any((d==code)&(c==349)) and any((d==code)&(c==1168))

@pytest.mark.parametrize('valid,call_count',[(0,1126),(1,1128),(256,1128),(257,1130),(3054,1150)])
def test_independent_budgets(valid,call_count):
    result=budgets(valid)
    assert result['head_onnx_calls']==call_count
    assert result['training_head_rows']==5000*(4209+valid)
    assert result['diagnostic_torch_rows']==287372+2*valid

def test_response_and_absolute_are_distinct():
    centers,p=synthetic()
    center_delta=np.ones((3,23))
    branch_delta=np.ones((2,23))
    result=physical_metrics(center_delta,branch_delta,centers,p,np.ones(23))
    assert result['nine_cell_response_objective']==0.
    assert result['nine_cell_absolute_branch_diagnostic']==pytest.approx(2/(99*9))

def test_total_base_change_included():
    centers,p=synthetic();p['base'][:]=2.
    result=physical_metrics(np.zeros((3,23)),np.full((2,23),-2.),centers,p,np.ones(23))
    assert result['nine_cell_response_objective']==0.
    assert result['nine_cell_absolute_branch_diagnostic']==0.

def test_teacher_response_and_span_used():
    centers,p=synthetic();p['target'][:]=3.
    result=physical_metrics(np.zeros((3,23)),np.ones((2,23)),centers,p,np.full(23,2.))
    assert result['nine_cell_response_objective']==pytest.approx(2/(99*9))

def test_first24_keeps_requested_failed_clocks():
    centers,p=synthetic()
    delta=np.stack([np.ones(23),np.full(23,4.)])
    result=physical_metrics(np.zeros((3,23)),delta,centers,p,np.ones(23))
    window=result['cells'][0]['first24_requested_window']
    assert window==dict(successor_control_range=[251,274],requested=24,valid=1,strict_failed=23,response_valid_MSE=1.)
    assert result['cells'][1]['first24_requested_window']['response_valid_MSE'] is None

def test_clipping_groups_do_not_drop_endpoints():
    centers,p=synthetic()
    result=physical_metrics(np.zeros((3,23)),np.stack([np.full(23,12.),np.ones(23)]),centers,p,np.ones(23))
    assert result['valid']==2 and result['cells'][0]['student_clipped_rows']==1
    group=result['groups']['policy_native_target_clipped']
    assert group==dict(valid_rows=1,response_MSE=144.,absolute_MSE=144.)

def test_empty_cell_zero_fixed_weight():
    centers,p=synthetic()
    result=physical_metrics(np.zeros((3,23)),np.ones((2,23)),centers,p,np.ones(23))
    assert result['cells'][1]['physical_response_requested_MSE']==0.
    assert result['cells'][1]['response_valid_MSE'] is None
    assert result['cells'][0]['effective_mass']==2/(9*99)

def test_proof_signed_zero_failure_preserved(tmp_path):
    proof=Proof(tmp_path)
    with pytest.raises(AssertionError):proof.exact(np.array([0.]),np.array([-0.]),'signed zero')
    proof.finish()
    with np.load(tmp_path/'first_mismatch.npz') as z:
        assert not np.signbit(z['actual'][0]) and np.signbit(z['expected'][0])

def test_tree_type_and_keys_strict(tmp_path):
    proof=Proof(tmp_path)
    proof.tree(dict(passed=True,values=[1.,2.]),dict(passed=True,values=[1.,2.]),'match')
    with pytest.raises(AssertionError):proof.tree(dict(value=1),dict(value=1.),'dtype')
    proof.finish()

def test_private_rng_does_not_advance_global():
    before=torch.get_rng_state().clone()
    private=torch.Generator(device='cpu')
    private.set_state(before)
    draws=torch.randint(0,99*23,(64,),generator=private)
    again=torch.Generator(device='cpu');again.set_state(before)
    assert torch.equal(draws,torch.randint(0,99*23,(64,),generator=again))
    assert torch.equal(torch.get_rng_state(),before)

def test_nominal_reduction_retains_original_float32_rounding():
    cells=np.ones((3,9),np.float64);cells[0,0]=1e10
    measured=logged_cell_reductions(cells)
    assert measured[0]==float(torch.from_numpy(cells[0].astype(np.float32)).mean())
    assert measured[0]!=float(np.mean(cells[0]))
    assert measured[1]==measured[2]==1.

def test_composition_keeps_two_left_associated_promoted_adds():
    n,v,p=np.array([1e16]),np.array([1.]),np.array([1.])
    assert np.array_equal(compose_three(n,v,p),(n+v)+p)
    assert not np.array_equal(compose_three(n,v,p),n+(v+p))

def test_cli_actual_receipt_required_and_no_trainer_import():
    source=(SOURCE/'audit_physical_fit_evidence.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    flags=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument']
    receipt=next(n for n in flags if n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--training-receipt-sha256')
    assert any(k.arg=='required' and isinstance(k.value,ast.Constant) and k.value.value is True for k in receipt.keywords)
    forbidden=('InferenceSession','make_actor','AdamW','mj_step','mj_forward','backward','forward','step')
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in calls)
    for name in ('audit_physical_fit_evidence.py','audit_physical_fit_math.py'):
        t=ast.parse((SOURCE/name).read_text(encoding='utf-8'))
        for n in ast.walk(t):
            if isinstance(n,ast.ImportFrom):assert not n.module.startswith(('fit_','chord_','physical_training_','physical_fit_diagnostics'))
    assert 'training_receipt_sha256' in source and 'final unchanged' in source
