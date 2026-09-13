"""Freeze one root-selected cost-ranked runtime after saved132 equivalence."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
PRIOR=NEW/'fast_controller_filtered_v1';DIAG=NEW/'filtered_all_candidates132_v1';EXPERT=NEW/'bfm_entry250_actual_oracle_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def local(value):
    value=str(value).replace('\\','/')
    return Path(value[5].upper()+':'+value[6:]) if value.startswith('/mnt/') else Path(value)

def main():
    assert not any((BASE/name).exists() for name in ('source_snapshot_v1','frozen_inputs_v2.json','runtime_clearance.json','nominal','canonical_process_status.json'))
    assert read(BASE/'root_selection.json')['selected'] and read(BASE/'root_selection.json')['canonical_trials']==1
    tests=('saved132_cost_equivalence_test_report.json','preservation_schema_test_report.json','cost_persistence_test_report.json')
    for name in tests:assert read(BASE/name)['pass_']
    for name in tests[1:]:assert read(BASE/name)['driver_sha256']==sha(BASE/'source_draft_v1/evaluate_filtered_student.py')
    for name,digest in read(BASE/tests[0])['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest
    previous=read(PRIOR/'frozen_inputs_v2.json');expert=read(EXPERT/'frozen_inputs.json')
    original={};differences={}
    for name,digest in previous['source_sha256'].items():
        current=sha(BASE/'source_draft_v1'/name)
        if current==digest:original[name]=current
        else:differences[name]=dict(previous_sha256=digest,current_sha256=current)
    changed={'evaluate_filtered_student.py','gear_sonic/utils/g1_true23_mjbatch_mpc.py',
        'gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py','gear_sonic/utils/g1_true23_mjbatch_model.py'}
    assert set(differences)==changed and len(original)==18
    for name in changed-{'evaluate_filtered_student.py'}:assert differences[name]['current_sha256']==expert['source_sha256'][name]
    assert sha(BASE/'source_draft_v1/native_forecast.py')=='81af01274f55c703850cb3ec3801bb41912e3f3786440af5099b769b7bd28824'
    inputs={}
    def bind(path,expected=None):
        path=local(path);actual=sha(path)
        if expected is not None:assert actual==expected,(str(path),actual,expected)
        key=path.as_posix()
        if key in inputs:assert inputs[key]==actual
        inputs[key]=actual
    for path,digest in read(DIAG/'request.json')['input_sha256'].items():bind(path,digest)
    for path,digest in read(BASE/tests[0])['input_sha256'].items():bind(path,digest)
    for path in (PRIOR/'frozen_inputs_v2.json',PRIOR/'nominal/trace.npz',PRIOR/'nominal/report.json',
        DIAG/'request.json',DIAG/'results/report.json',DIAG/'results/forecasts.npz',DIAG/'results/cost_knots.npz',
        EXPERT/'frozen_inputs.json',BASE/'root_selection.json',BASE/'run_canonical_durable.ps1',Path(__file__),
        BASE/'saved132_test_initial_constructor_fault.json',PRIOR/'transactional_stub_test_report_v2.json',
        PRIOR/'recorded_fixture_test_report.json'):
        bind(path)
    for name in tests:bind(BASE/name)
    for path in sorted((BASE/'tests').glob('*.py')):bind(path)
    draft=BASE/'source_draft_v1';snapshot=BASE/'source_snapshot_v1';sources={}
    for path in sorted(draft.rglob('*.py')):
        ast.parse(path.read_text());relative=path.relative_to(draft);dest=snapshot/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);sources[relative.as_posix()]=sha(dest)
    assert len(sources)==26
    receipt=dict(kind='one_root_selected_cost_ranked65000_runtime',source_directory='source_snapshot_v1',source_sha256=sources,input_sha256=inputs,
        original_filtered_sources_byteexact=original,changed_existing_sources=differences,
        added_sources=sorted(set(sources)-set(previous['source_sha256'])),qualified_cost_dependencies_exact=True,
        canonical_trials=1,nominal_controls=1569,conditional_continuous_hold_controls=250,
        source_review_pending=True,root_selection_pending=False,selection='lowest_exact_feasible_prefix_cost',
        evaluate_all_four=True,candidate_order=['primary','original_BFM','previous_applied','current_q'],
        exact_ties='first in original order',private_horizon_controls=5,actual_commit_controls=1,
        initial_BFM_controls=250,terminal_BFM_yaw4_start=1269,
        preserved_history='raw initial BFM; moving applied-target normalization; unchanged primary terminal BFM raw action, otherwise selected-target normalization',
        ordinary_final_step=65000,no_weight_reference_physics_horizon_changes=True,optimizer_calls=0,new_labels=0,
        saved132_equivalence_passed=True,prelaunch_connected_trials=0,canonical_trials_started=0,hardware_authorized=False)
    write(BASE/'frozen_inputs_v2.json',receipt)
    bindings=[BASE/'frozen_inputs_v2.json',BASE/'root_selection.json',BASE/'run_canonical_durable.ps1',
        snapshot/'evaluate_filtered_student.py',snapshot/'native_forecast.py',snapshot/'transactional_student.py',
        snapshot/'ranked_admission.py',snapshot/'prefix_tracking_cost.py',BASE/'saved132_cost_equivalence_test_report.json',
        NEW/'fast_controller_phase_fit_v1/fit/student_head.onnx',DIAG/'results/report.json']
    clearance=dict(kind='cost_ranked_runtime_clearance_draft',canonical_rollout_authorized=False,canonical_trials=1,
        root_selection_pending=False,source_review_pending=True,selection='lowest_exact_feasible_prefix_cost',evaluate_all_four=True,
        private_horizon_controls=5,actual_commit_controls=1,head_sha256=sha(NEW/'fast_controller_phase_fit_v1/fit/student_head.onnx'),
        frozen_sources_sha256=sha(BASE/'frozen_inputs_v2.json'),
        bound_files=[dict(path=str(path),sha256=sha(path)) for path in bindings],optimizer_calls=0,new_fitting=False,new_expert_query=False,hardware_authorized=False)
    write(BASE/'runtime_clearance_DRAFT.json',clearance)
    status=dict(frozen_receipt_sha256=sha(BASE/'frozen_inputs_v2.json'),sources=len(sources),inputs=len(inputs),
        driver_sha256=sources['evaluate_filtered_student.py'],cost_sha256=sources['prefix_tracking_cost.py'],
        admission_sha256=sources['ranked_admission.py'],launcher_sha256=sha(BASE/'run_canonical_durable.ps1'),
        saved132_equivalence_sha256=sha(BASE/'saved132_cost_equivalence_test_report.json'),source_review_pending=True,no_canonical_started=True)
    write(BASE/'frozen_preparation_status.json',status);print(json.dumps(status))

if __name__=='__main__':main()
