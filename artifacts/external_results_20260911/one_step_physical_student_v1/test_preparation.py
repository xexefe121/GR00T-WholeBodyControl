"""Tiny synthetic tensors/stub sessions and source checks; no actual models or fit."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_draft_v1'
sys.path.insert(0,str(SOURCE))
from physical_training_objective import physical_response_loss,physical_cells,expected_budgets
from physical_training_inputs import expected_identities,validate_status_vectors,resolve_array_path,validate_bound_review,sha
from physical_fit_diagnostics import evaluate_physical,sync_counts

def small_case():
    centers=torch.zeros((3,23),dtype=torch.float32,requires_grad=True)
    branches=torch.ones((2,23),dtype=torch.float32,requires_grad=True)
    successor=torch.tensor([1,2],dtype=torch.int64)
    base=torch.zeros((3,23),dtype=torch.float64)
    bbase=torch.zeros((2,23),dtype=torch.float64)
    cells=[np.array([0],np.int64),np.array([1],np.int64)]+[np.empty(0,np.int64) for _ in range(7)]
    args=(centers,branches,successor,base,bbase,base.clone(),bbase.clone(),torch.ones(23),cells)
    return args

def test_live_successor_gradient_and_requested_denominator():
    args=small_case();loss,cells=physical_response_loss(*args)
    assert float(loss.detach())==pytest.approx((1/99+1/819)/9)
    assert float(cells[0].detach())==pytest.approx(1/99) and float(cells[1].detach())==pytest.approx(1/819)
    loss.backward()
    assert torch.count_nonzero(args[0].grad[0])==0
    torch.testing.assert_close(args[0].grad[1],torch.full((23,),-2/(9*99*23)),rtol=1e-6,atol=1e-10)
    torch.testing.assert_close(args[1].grad[1],torch.full((23,),2/(9*819*23)),rtol=1e-6,atol=1e-10)

def test_nonzero_base_response_corrected_by_residual():
    args=list(small_case());args[4]=torch.full((2,23),2.,dtype=torch.float64)
    args[1]=torch.full((2,23),-2.,requires_grad=True)
    loss,_=physical_response_loss(*args)
    assert float(loss.detach())==0

def test_matching_teacher_total_response_is_zero():
    args=list(small_case());args[6]=torch.ones((2,23),dtype=torch.float64)
    loss,_=physical_response_loss(*args);assert float(loss.detach())==0

def test_equal_bias_not_hidden_from_nominal_anchor():
    args=list(small_case());args[0]=torch.ones((3,23),requires_grad=True)
    loss,_=physical_response_loss(*args);assert float(loss.detach())==0
    assert float((args[0]**2).mean().detach())==1

def test_empty_cells_do_not_redistribute_weight():
    args=list(small_case());args[8]=[np.empty(0,np.int64) for _ in range(9)]
    loss,cells=physical_response_loss(*args);assert float(loss.detach())==0 and torch.isfinite(cells).all()
    loss.backward();assert torch.count_nonzero(args[0].grad)==0

def test_exact_clock_phase_metadata():
    d,c,index=expected_identities();cells=physical_cells(d,c+1)
    assert [len(ids) for ids in cells]==[99,819,100]*3
    for code in range(3):
        rows=np.flatnonzero(d==code)
        assert np.array_equal(c[rows],np.arange(250,1268))
        assert np.array_equal(index[rows]+1,code*1019+np.arange(1,1019))
    assert all(np.any((c==boundary)&(d==code)) for code in range(3) for boundary in (349,1168,1267))

@pytest.mark.parametrize('valid,expected',[(0,1126),(1,1128),(256,1128),(257,1130),(3054,1150)])
def test_exact_call_and_row_budgets(valid,expected):
    b=expected_budgets(valid)
    assert b['head_onnx_calls']==expected
    assert b['training_head_rows']==5000*(4209+valid)
    assert b['training_head_rows']<=36315000
    assert b['diagnostic_torch_rows']<=293480

@pytest.mark.parametrize('value',[-1,3055,2.5])
def test_invalid_budget_rejected(value):
    with pytest.raises(AssertionError):expected_budgets(value)

def test_failed_status_explicit():
    validate_status_vectors(np.array([True,False]),np.ones(2,bool),np.ones(2,np.int32),np.array([1,2],np.int32))

@pytest.mark.parametrize('status',[0,1,3])
def test_failed_unattempted_exception_or_hidden_valid_rejected(status):
    with pytest.raises(AssertionError):
        validate_status_vectors(np.array([False]),np.array([True]),np.array([1]),np.array([status]))

def test_nominal_mismatch_rejected():
    with pytest.raises(AssertionError):
        validate_status_vectors(np.array([True]),np.array([False]),np.array([1]),np.array([1]))

def test_manifest_path_cannot_escape(tmp_path):
    assert resolve_array_path(tmp_path/'manifest.json','x.npy')==tmp_path/'x.npy'
    with pytest.raises(AssertionError):resolve_array_path(tmp_path/'manifest.json','../x.npy')

def test_direct_review_subject_and_pass_gate(tmp_path):
    data=tmp_path/'x.bin';data.write_bytes(b'existing-synthetic-input')
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(passed=True,subject=sha(data))))
    pins={str(data):sha(data),str(review):sha(review)}
    spec=dict(path=str(review),sha256=sha(review),pass_field='passed',subjects={str(data):sha(data)})
    validate_bound_review(spec,pins)
    review.write_text(json.dumps(dict(passed=True,subject='wrong')))
    pins[str(review)]=sha(review);spec['sha256']=sha(review)
    with pytest.raises(AssertionError):validate_bound_review(spec,pins)

def stub_ledger():
    return dict(head_onnx_calls_attempted=0,head_onnx_calls_returned=0,
        diagnostic_torch_rows_attempted=0,diagnostic_torch_rows_returned=0)

def stub_actor(x):
    # Not a model: shape-only zeros for accounting/failure tests.
    return torch.zeros((len(x),23),dtype=torch.float32)

class StubSession:
    def run(self,outputs,feed):return [np.zeros((len(feed['features']),23),np.float32)]

def test_stub_diagnostics_fixed_calls(tmp_path):
    ledger=stub_ledger();x=np.zeros((257,1069),np.float32)
    result=evaluate_physical(stub_actor,x,np.zeros(1069,np.float32),np.ones(1069,np.float32),np.ones(23,np.float32),
        'no-model.onnx',ledger,tmp_path,'stub',4,session_factory=lambda _:StubSession())
    assert result['onnx_calls']==2 and result['export_parity_passed']
    assert ledger['head_onnx_calls_attempted']==ledger['head_onnx_calls_returned']==2
    assert ledger['diagnostic_torch_rows_attempted']==ledger['diagnostic_torch_rows_returned']==257

def test_stub_ort_exception_preserves_attempted_inputs(tmp_path):
    class Fault:
        def run(self,*args):raise RuntimeError('synthetic fault')
    ledger=stub_ledger();x=np.zeros((2,1069),np.float32)
    with pytest.raises(RuntimeError):
        evaluate_physical(stub_actor,x,np.zeros(1069,np.float32),np.ones(1069,np.float32),np.ones(23,np.float32),
            'no-model.onnx',ledger,tmp_path,'failed',2,session_factory=lambda _:Fault())
    assert ledger['head_onnx_calls_attempted']==1 and ledger['head_onnx_calls_returned']==0
    with np.load(tmp_path/'failed_failed_physical_diagnostics.npz') as z:
        assert np.array_equal(z['current_input'],x) and z['valid_torch'].all() and not z['valid_onnx'].any()

def test_stub_nonfinite_return_preserved(tmp_path):
    class Fault:
        def run(self,_,feed):return [np.full((len(feed['features']),23),np.nan,np.float32)]
    ledger=stub_ledger()
    with pytest.raises(AssertionError):
        evaluate_physical(stub_actor,np.zeros((1,1069),np.float32),np.zeros(1069,np.float32),np.ones(1069,np.float32),np.ones(23,np.float32),
            'no-model.onnx',ledger,tmp_path,'nan',2,session_factory=lambda _:Fault())
    assert ledger['head_onnx_calls_returned']==1
    with np.load(tmp_path/'nan_failed_physical_diagnostics.npz') as z:assert np.isnan(z['last_returned_output']).all()

def test_stub_budget_prevents_extra_call(tmp_path):
    ledger=stub_ledger();ledger['head_onnx_calls_attempted']=2
    with pytest.raises(AssertionError):
        evaluate_physical(stub_actor,np.zeros((1,1069),np.float32),np.zeros(1069,np.float32),np.ones(1069,np.float32),np.ones(23,np.float32),
            'no-model.onnx',ledger,tmp_path,'budget',2,session_factory=lambda _:StubSession())
    assert ledger['head_onnx_calls_attempted']==2

def test_stub_multiple_outputs_all_preserved(tmp_path):
    class Fault:
        def run(self,_,feed):return [np.zeros((len(feed['features']),23),np.float32),np.ones((1,2),np.float32)]
    with pytest.raises(AssertionError):
        evaluate_physical(stub_actor,np.zeros((1,1069),np.float32),np.zeros(1069,np.float32),np.ones(1069,np.float32),np.ones(23,np.float32),
            'no-model.onnx',stub_ledger(),tmp_path,'extra',2,session_factory=lambda _:Fault())
    with np.load(tmp_path/'extra_failed_physical_diagnostics.npz') as z:
        assert np.array_equal(z['last_returned_output_1'],np.ones((1,2),np.float32))

def test_separate_original_counters_sum_without_weakening():
    ledger=dict(legacy_diagnostics=stub_ledger(),physical_diagnostics=stub_ledger())
    ledger['legacy_diagnostics']['head_onnx_calls_attempted']=1126
    ledger['physical_diagnostics']['head_onnx_calls_attempted']=24
    sync_counts(ledger);assert ledger['head_onnx_calls_attempted']==1150

def test_no_actual_request_or_run_exists():
    for name in ('training_request.json','training_frozen_inputs.json','training_clearance.json','fit'):
        assert not (BASE/name).exists()

def test_preserved_originals_and_exact_derivation():
    report=json.loads((BASE/'source_derivation.json').read_text())
    for name,digest in report['original_sources_sha256'].items():assert sha(SOURCE/name)==digest
    original=(BASE.parent/'velocity_chord_student_v1/source_snapshot_v3/fit_velocity_chords.py').read_text(encoding='utf-8')
    for edit in report['exact_substitutions']:
        assert original.count(edit['old'])==edit['count'];original=original.replace(edit['old'],edit['new'])
    assert original==(SOURCE/'fit_physical_continuation.py').read_text(encoding='utf-8')

def test_main_has_preflight_before_any_model_construction():
    tree=ast.parse((SOURCE/'fit_physical_continuation.py').read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    calls=[(n.lineno,ast.unparse(n.func)) for n in ast.walk(main) if isinstance(n,ast.Call)]
    gate=min(line for line,name in calls if name=='check_inputs')
    load=min(line for line,name in calls if name=='load_physical')
    model=min(line for line,name in calls if name=='make_actor')
    assert gate<load<model
    assert sum(name=='optimizer.step' for _,name in calls)==1
    assert sum(name=='sample_pairs' for _,name in calls)==1
    assert not any(name.endswith(('mj_step','mj_forward','manual_seed')) for _,name in calls)
