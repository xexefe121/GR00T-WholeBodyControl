"""Pure/stub checks: no native dynamics or ONNX session construction."""
import ast
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest

BASE=Path(__file__).parent
SOURCE=BASE/'source_snapshot_v1'
sys.path.insert(0,str(SOURCE))
from proposal_evidence import ProposalEvidence,RecordingSession,exact,trace_arrays,TRAILING_SHAPES
from evaluation_gate import bound_file,require_ready,sha
from head_activation_witness import require_witness_ready,CENTERS_SHA,LABEL_SHA


def function(path,name,namespace):
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def test_original_eighteen_preserved():
    old=BASE.parent/'fast_controller_phase_fit_v1/source_snapshot_v1'
    files=list(old.rglob('*.py'));assert len(files)==18
    for path in files:assert path.read_bytes()==(SOURCE/path.relative_to(old)).read_bytes()


@pytest.mark.parametrize('name',['get_state','assess','new_trace','source_metrics','zero_parity'])
def test_unchanged_original_functions(name):
    def node(path):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(node(SOURCE/'evaluate_phase_student.py'))==ast.dump(node(SOURCE/'evaluate_velocity_chord_student.py'))


def test_derivation_receipt_exact():
    record=json.loads((BASE/'evaluator_derivation.json').read_text())
    original=SOURCE/'evaluate_phase_student.py';text=original.read_text()
    for change in record['substitutions']:
        assert text.count(change['old'])==1;text=text.replace(change['old'],change['new'])
    assert text==(SOURCE/'evaluate_velocity_chord_student.py').read_text()
    assert record['original_sha256']==sha(original)
    assert record['derived_sha256']==sha(SOURCE/'evaluate_velocity_chord_student.py')


def test_recording_session_passes_original_objects_once():
    features=np.asarray([[-0.,3]],np.float32);returned=[np.asarray([[2.]],np.float32)]
    calls=[]
    def run(names,feed):calls.append((names,feed));return returned
    evidence=SimpleNamespace(current={});session=RecordingSession(SimpleNamespace(run=run),'head',evidence)
    feed={'features':features};result=session.run(None,feed)
    assert result is returned and len(calls)==1 and calls[0][1] is feed
    assert exact(evidence.current['head_input_features'],features)
    assert exact(evidence.current['head_output_0'],returned[0])


def test_recording_session_retains_inputs_on_fault():
    def fail(*args):raise RuntimeError('stub ORT fault')
    evidence=SimpleNamespace(current={});session=RecordingSession(SimpleNamespace(run=fail),'actor',evidence)
    value=np.asarray([[-0.,2]],np.float32)
    with pytest.raises(RuntimeError):session.run(None,{'state':value})
    assert exact(evidence.current['actor_input_state'],value)


def test_nonfinite_return_preserved_before_validation():
    output=[np.full((1,23),np.nan,np.float32)]
    evidence=SimpleNamespace(current={})
    wrapped=RecordingSession(SimpleNamespace(run=lambda *a:output),'head',evidence)
    assert wrapped.run(None,{'features':np.zeros((1,1069),np.float32)}) is output
    assert exact(evidence.current['head_output_0'],output[0])


class History:
    def __init__(self):self.data={'x':np.zeros((4,75),np.float32)}
    def before_update(self,terms):
        old=self.data['x'].reshape(-1).copy();self.data['x'][1:]=self.data['x'][:-1];self.data['x'][0]=terms
        return old


def make_runtime():
    c={'joint_limits':np.tile([-1.,1.],(23,1)),'default_q':np.zeros(23),
       'kp':np.ones(23)*2,'training_effort':np.ones(23)*8}
    seed=SimpleNamespace(recorded_controls=250,previous_action=np.zeros(23,np.float32),history=History(),
        sessions={'actor':SimpleNamespace(run=lambda *a:None),'backward':SimpleNamespace(run=lambda *a:None)},
        _terms=lambda q,dq,p:(np.zeros(52,np.float32),np.ones(75,np.float32)))
    runtime=SimpleNamespace(seed=seed,c=c,limits=c['joint_limits'],features=lambda *a:np.zeros(1069,np.float32),
        head=SimpleNamespace(run=lambda *a:[np.full((1,23),3,np.float32)]))
    def base(*a):return np.full(23,2,np.float32),np.full(23,2.),np.zeros(52,np.float32)
    runtime.propose=function(SOURCE/'student_linear_runtime.py','propose',dict(np=np,time=time,infer_base=base)).__get__(runtime)
    return runtime,c


def test_raw_prior_preserved_under_target_clipping():
    left,c=make_runtime();right,_=make_runtime();evidence=ProposalEvidence(right,c)
    evidence.begin(250,np.zeros(30),np.zeros(29),np.zeros(291),5.,np.zeros(8,np.int32),np.zeros(8,np.int32))
    a=left.propose(250,np.zeros(30),np.zeros(29));b=right.propose(250,np.zeros(30),np.zeros(29))
    evidence.accepted_proposal(b)
    for key in a:
        if key!='inference_ms':assert exact(a[key],b[key]),key
    assert exact(left.seed.previous_action,right.seed.previous_action)
    assert exact(left.seed.history.data['x'],right.seed.history.data['x'])
    assert np.all(b['target']==1) and np.all(b['action']==5)
    assert np.all(evidence.last_raw_proposal==5) and np.all(evidence.last_actual_action==1)


def test_preserve_before_after_history_and_rejected_output(tmp_path):
    runtime,c=make_runtime();evidence=ProposalEvidence(runtime,c)
    evidence.begin(250,np.zeros(30),np.zeros(29),np.arange(291.),5.,np.zeros(8,np.int32),np.zeros(8,np.int32))
    proposed=runtime.propose(250,np.zeros(30),np.zeros(29));evidence.accepted_proposal(proposed)
    path=tmp_path/'rejected.npz';evidence.preserve(path,'parity',np.zeros(30),np.zeros(29),np.arange(291.),5.,np.zeros(8,np.int32),np.zeros(8,np.int32))
    with np.load(path) as saved:
        assert saved['recorded_controls_before']==250 and saved['recorded_controls_after']==251
        assert np.all(saved['previous_action_before']==0) and np.all(saved['previous_action_after']==5)
        assert exact(saved['head_output_0'],np.full((1,23),3,np.float32))
        assert exact(saved['integration_before'],np.arange(291.))


def empty_trace():
    data=SimpleNamespace(qpos=np.zeros(30),qvel=np.zeros(29),time=0.,warning=SimpleNamespace(number=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32)))
    trace=function(SOURCE/'evaluate_velocity_chord_student.py','new_trace',{})(data)
    trace.update(raw_proposal=[],actual_normalized_action=[])
    return trace


def test_immediate_rejection_empty_schema():
    trace=empty_trace()
    trace['control_integration_before']=[np.zeros(291)];trace['control_history_before']=[np.zeros(300,np.float32)]
    trace['control_previous_action_before']=[np.zeros(23,np.float32)]
    arrays=trace_arrays(trace)
    for key,shape in TRAILING_SHAPES.items():assert arrays[key].shape[1:]==shape,key
    assert arrays['features'].dtype==np.float32 and arrays['global_control'].dtype==np.int64
    assert arrays['target'].shape==(0,23)


def test_trace_rejects_inconsistent_physics_count():
    trace=empty_trace();trace['physics_torque']=[np.zeros(23)]
    with pytest.raises(ValueError,match='Native sample'):trace_arrays(trace)


def test_absent_bindings_fail_before_any_model(tmp_path):
    with pytest.raises(ValueError,match='unbound'):require_ready(tmp_path)
    with pytest.raises(ValueError,match='not bound'):require_witness_ready(tmp_path)


@pytest.mark.parametrize('entry',[{}, {'path':'fake','sha256':'pending'}, {'path':'relative','sha256':'0'*64}])
def test_placeholder_bindings_rejected(entry):
    with pytest.raises(ValueError):bound_file(entry)


def test_exact_means_dtype_shape_and_signed_zero():
    assert not exact(np.asarray([-0.],np.float32),np.asarray([0.],np.float32))
    assert not exact(np.zeros(1,np.float32),np.zeros(1,np.float64))
    assert not exact(np.zeros(1,np.float32),np.zeros((1,1),np.float32))


def test_fixed_witness_row_matches_original_query_without_inference():
    centers=BASE.parent/'velocity_chord_student_v1/generation/centers.npz'
    labels=BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz'
    assert sha(centers)==CENTERS_SHA and sha(labels)==LABEL_SHA
    with np.load(centers) as c,np.load(labels) as l:
        assert c['dataset'][2038]==2 and c['control'][2038]==250 and c['source_frame'][2038]==261
        for key in ('features','base_target','previous_action','history','state'):
            assert exact(c[key][2038],l[key][0]),key
        assert exact(c['qpos'][2038],l['teacher_qpos'][0]) and exact(c['qvel'][2038],l['teacher_qvel'][0])


def test_witness_single_head_call_ast_and_no_dynamics():
    text=(SOURCE/'head_activation_witness.py').read_text();tree=ast.parse(text)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='head' and n.func.attr=='run']
    assert len(calls)==1
    assert 'mj_step' not in text and 'infer_base' not in text
    assert text.index('attempted=1')<text.index('returned=head.run')
    assert text.index("context['returned_'")<text.index('assert len(returned)')


def test_native_exception_preserves_completed_samples_and_unadvanced_clock(tmp_path):
    runtime,c=make_runtime();evidence=ProposalEvidence(runtime,c)
    before=np.arange(291.);warnings=np.zeros(8,np.int32)
    evidence.begin(250,np.zeros(30),np.zeros(29),before,5.002,warnings,warnings)
    trace=empty_trace();trace['physics_torque']=[np.ones(23)]
    data=SimpleNamespace(qpos=np.zeros(30),qvel=np.zeros(29),ctrl=np.ones(23)*2,
                         warning=SimpleNamespace(number=warnings,lastinfo=warnings))
    def fault(*args):raise RuntimeError('stub native exception')
    tree=ast.parse((SOURCE/'evaluate_velocity_chord_student.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Try) and len(n.body)==1 and
        isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Call) and
        isinstance(n.body[0].value.func,ast.Attribute) and n.body[0].value.func.attr=='mj_step')
    environment=dict(np=np,json=json,mujoco=SimpleNamespace(mj_step=fault),native=None,data=data,evidence=evidence,
        get_state=lambda *a:before.copy(),dest=tmp_path,expected_time=5.002,sub=1,control=250,
        actual_substeps=1,count=1569,trace=trace)
    with pytest.raises(RuntimeError,match='stub native exception'):
        exec(compile(ast.Module(body=[node],type_ignores=[]),'native_exception_stub','exec'),environment)
    assert environment['expected_time']==5.002 and environment['actual_substeps']==1
    with np.load(tmp_path/'native_exception_state.npz') as saved:
        assert saved['expected_time_at_failure']==5.002
        assert exact(saved['attempted_native_ctrl'],data.ctrl)
        assert saved['native_step_attempt_substep']==1
    with np.load(tmp_path/'native_exception_completed_samples.npz') as saved:
        assert exact(saved['physics_torque'],np.ones((1,23)))
    report=json.loads((tmp_path/'fatal_failure.json').read_text())
    assert report['unclassified_attempted_step'] and not report['full_segment_completed']
    assert report['requested_controls']==1569


def test_assess_preserves_clock_and_exact_speed_boundary():
    assess=function(SOURCE/'evaluate_velocity_chord_student.py','assess',dict(np=np))
    q=np.zeros(30);q[2]=.8;q[3]=1
    dq=np.zeros(29);dq[6]=20
    data=SimpleNamespace(qpos=q,qvel=dq,qfrc_actuator=np.zeros(29),warning=SimpleNamespace(number=np.zeros(8)),time=5.002)
    c=dict(joint_limits=np.tile([-1.,1.],(23,1)),native_velocity=np.ones(23)*20,native_effort=np.ones(23))
    assert assess(data,c,5.002)[0]==[]
    dq[6]=np.nextafter(20.,np.inf)
    assert 'native_joint_speed' in assess(data,c,5.002)[0]
    assert 'physics_clock' in assess(data,c,5.004)[0]
