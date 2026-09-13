"""Freeze prepared controller after component parity. Does not authorize execution."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
PRIOR=NEW/'fast_controller_phase_fit_v1';COMPONENT=NEW/'phase_student_preallocated_forecast_v2'
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    assert not (BASE/'source_snapshot_v1').exists() and not (BASE/'frozen_inputs_v2.json').exists()
    assert not (BASE/'nominal').exists() and not (BASE/'runtime_clearance.json').exists()
    parity=read(COMPONENT/'results/report.json')
    assert parity['pass_'] and parity['cases']==232 and parity['private_component_steps']==6480
    assert parity['all_state_torque_force_time_warning_samples_bitexact'] and parity['all_caller_integration_and_warnings_unchanged']
    assert sha(COMPONENT/'results/report.json')=='67e516f3cfd059438677580b8a7b575b183af270327f3143a13748b62201e478'
    assert sha(BASE/'source_draft_v1/native_forecast.py')==sha(COMPONENT/'native_forecast.py')=='81af01274f55c703850cb3ec3801bb41912e3f3786440af5099b769b7bd28824'
    previous=read(PRIOR/'frozen_inputs_v2.json')
    for name,expected in previous['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==expected
    test_reports=('transactional_stub_test_report_v2.json','ordered_admission_stub_test_report.json','recorded_fixture_test_report.json','preservation_schema_test_report.json')
    for name in test_reports:
        assert read(BASE/name)['pass_']
    assert read(BASE/'preservation_schema_test_report.json')['driver_sha256']==sha(BASE/'source_draft_v1/evaluate_filtered_student.py')
    assert read(BASE/'root_selection.json')['canonical_trials']==1 and read(BASE/'root_selection.json')['selected'] is True
    proposal=ROOT/'artifacts/teleop_resume_20260911/PROPOSED_FILTER_EXPERIMENT.md'
    shutil.copyfile(proposal,BASE/'proposal_snapshot.md')
    inputs={}
    def bind(path,expected=None):
        path=Path(path);key=path.as_posix();actual=sha(path)
        if expected is not None:assert actual==expected,(path,actual,expected)
        if key in inputs:assert inputs[key]==actual
        inputs[key]=actual
    bind(PRIOR/'frozen_inputs_v2.json')
    for path,expected in previous['input_sha256'].items():bind(path,expected)
    for path in [PRIOR/name for name in ('fit/student_head.onnx','fit/student_head.pt','fit/report.json','fit/teacher_fit.npz',
        'final_export_validation.json','final_export_clearance.json','nominal/trace.npz','nominal/report.json')]+[
        COMPONENT/'native_forecast.py',COMPONENT/'request.json',COMPONENT/'cases.npz',COMPONENT/'results/report.json',
        COMPONENT/'results/preallocated_traces.npz',COMPONENT/'results/oracle_traces.npz',COMPONENT/'run_parity.py',
        COMPONENT/'oracle_snapshot.py',proposal,BASE/'proposal_snapshot.md',BASE/'run_canonical_durable.ps1',BASE/'root_selection.json',Path(__file__),
        NEW/'phase_student_failure_diagnosis_v1/arrays.npz',NEW/'phase_student_failure_diagnosis_v1/report.json',
        NEW/'three_expert_feedback_audit_v1/report.json',NEW/'three_expert_feedback_audit_v1/README.md']:
        bind(path)
    for path in sorted((BASE/'tests').glob('*.py')):bind(path)
    for name in test_reports:bind(BASE/name)
    snapshot=BASE/'source_snapshot_v1'
    sourcefiles=sorted((BASE/'source_draft_v1').rglob('*.py'))
    assert len(sourcefiles)==22
    for path in sourcefiles:
        ast.parse(path.read_text(encoding='utf-8'))
        dest=snapshot/path.relative_to(BASE/'source_draft_v1');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    sources={path.relative_to(snapshot).as_posix():sha(path) for path in sorted(snapshot.rglob('*.py'))}
    receipt=dict(kind='prepared_fixed65000_private_admission_runtime',source_directory='source_snapshot_v1',
        source_sha256=sources,input_sha256=inputs,original18_phase_sources_byteexact=True,
        preview_component_sha256=sha(COMPONENT/'native_forecast.py'),preview_parity_report_sha256=sha(COMPONENT/'results/report.json'),
        candidate_order=['primary','original_BFM','previous_applied','current_q'],private_horizon_controls=5,actual_commit_controls=1,
        canonical_trials=1,root_selection_pending=False,source_review_pending=True,
        actual_inference_calls_started=0,actual_physics_steps_started=0,optimizer_updates=0,hardware_authorized=False)
    receiptpath=BASE/'frozen_inputs_v2.json';write(receiptpath,receipt)
    bindings=[receiptpath,BASE/'run_canonical_durable.ps1',BASE/'proposal_snapshot.md',
        snapshot/'evaluate_filtered_student.py',snapshot/'native_forecast.py',snapshot/'transactional_student.py',snapshot/'ordered_admission.py',
        COMPONENT/'results/report.json',PRIOR/'fit/student_head.onnx',PRIOR/'fit/report.json',BASE/'root_selection.json']
    draft=dict(kind='filtered_runtime_clearance_draft',canonical_rollout_authorized=False,canonical_trials=1,
        root_selection_pending=False,source_review_pending=True,head_sha256=sha(PRIOR/'fit/student_head.onnx'),
        frozen_sources_sha256=sha(receiptpath),private_horizon_controls=5,actual_commit_controls=1,
        bound_files=[dict(path=str(path),sha256=sha(path)) for path in bindings],
        no_new_fitting=True,no_new_expert_query=True,hardware_authorized=False)
    write(BASE/'runtime_clearance_DRAFT.json',draft)
    status=dict(frozen_receipt_sha256=sha(receiptpath),sources=len(sources),inputs=len(inputs),
        driver_sha256=sources['evaluate_filtered_student.py'],transaction_sha256=sources['transactional_student.py'],
        admission_sha256=sources['ordered_admission.py'],preview_sha256=sources['native_forecast.py'],
        launcher_sha256=sha(BASE/'run_canonical_durable.ps1'),root_selection_pending=False,source_review_pending=True,no_execution_started=True)
    write(BASE/'frozen_preparation_status.json',status);print(json.dumps(status))

if __name__=='__main__':main()
