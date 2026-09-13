"""Freeze already-authorized phase fit sources and audited inputs; no inference."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
PRIOR=NEW/'fast_controller_aggregate_fit_v1'
LABEL=NEW/'bfm_entry250_labels_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    receipt=BASE/'frozen_inputs_v2.json';snapshot=BASE/'source_snapshot_v1'
    assert not receipt.exists() and not snapshot.exists() and not (BASE/'fit').exists()
    auditpath=NEW/'bfm250_expert_labels_independent_v1/report.json';audit=read(auditpath)
    assert audit['independent_saved_array_label_checks_pass'] and audit['samples']==1019
    assert audit['total_phase_only_training_rows']==3057 and audit['exact_duplicate_groups']==audit['exact_conflicts']==0
    assert audit['maximum_distance_difference']==0 and audit['all_actual_targets_histories_prior_integration_bitexact']
    compatibility=read(LABEL/'compatibility/report.json')
    assert compatibility['compatibility_pass'] and compatibility['total_samples']==3057
    inputs={}
    def bind(path,expected=None):
        path=Path(path);key=path.as_posix();actual=sha(path)
        if expected is not None:assert actual==expected,(path,actual,expected)
        if key in inputs:assert inputs[key]==actual
        inputs[key]=actual
    for receiptpath in (PRIOR/'frozen_inputs_v2.json',LABEL/'collector_frozen_inputs.json'):
        prior=read(receiptpath);bind(receiptpath)
        for path,expected in prior['input_sha256'].items():bind(path,expected)
    for path,expected in audit['hashes'].items():bind(path,expected)
    for path in [auditpath,NEW/'phase_only_student_experiment_review_v1/proposal.json',
        NEW/'phase_only_student_experiment_review_v1/README_v2.md',
        NEW/'bfm250_collector_prelaunch_review_v1/review.json',LABEL/'labels/labels.npz',
        LABEL/'labels/report.json',LABEL/'labels/existing_normalization.npz',LABEL/'compatibility/report.json',
        LABEL/'compatibility/distances_and_pairs.npz',LABEL/'compatibility/fixed_pairs.json',
        LABEL/'compatibility/exact_duplicate_groups.json',LABEL/'launch_receipt.json',LABEL/'process_status.json',
        PRIOR/'fit/student_head.pt',PRIOR/'fit/student_head.onnx',PRIOR/'fit/teacher_fit.npz',PRIOR/'fit/report.json',
        PRIOR/'fit/optimization_completed.json',PRIOR/'fit/restoration20000_parity.json',
        PRIOR/'final_export_validation.json',PRIOR/'final_export_clearance.json',
        NEW/'aggregate_student_final_export_review_v1/review.json',
        NEW/'original_bfm_entry250_v1/entry250/trace.npz',NEW/'original_bfm_entry250_v1/entry250/report.json',
        NEW/'original_bfm_entry250_independent_physics_v1/report.json',NEW/'original_bfm_entry250_independent_intent_v1/report.json',
        BASE/'run_fit_durable.ps1',BASE/'run_canonical_durable.ps1',BASE/'validate_ordinary_final.py',Path(__file__)]:bind(path)
    original=read(PRIOR/'frozen_inputs_v2.json')['source_sha256']
    for name,expected in original.items():assert sha(BASE/'source_draft_v1'/name)==expected
    files=[p for p in (BASE/'source_draft_v1').rglob('*') if p.is_file()]
    assert len(files)==18
    for path in files:ast.parse(path.read_text(encoding='utf-8'))
    ast.parse((BASE/'validate_ordinary_final.py').read_text(encoding='utf-8'))
    shutil.copytree(BASE/'source_draft_v1',snapshot)
    sources={p.relative_to(snapshot).as_posix():sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    result=dict(kind='one_phase_only_fullbatch_student_fit_and_canonical_runtime',source_directory='source_snapshot_v1',
        source_sha256=sources,input_sha256=inputs,previous_global_step=60000,ordinary_final_global_step=65000,
        additional_updates=5000,full_batch_rows=3057,phase_rows_each=[100,819,100],equal_dataset_phase_cells=9,
        normalization_refitted=False,full_model_AdamW_RNG_restoration=True,
        learning_rate='3e-6+0.5*(3e-5-3e-6)*(1+cos(pi*k/4999)), k=0..4999',
        root_data_audit_sha256=sha(auditpath),compatibility_report_sha256=sha(LABEL/'compatibility/report.json'),
        fit_authorized_after_review=True,canonical_eval_authorized_only_after_final_export_review=True,
        original16_sources_byteexact=True,source_preparation_inference_calls=0,source_preparation_optimizer_calls=0,
        source_preparation_physics_steps=0,hardware_authorized=False)
    write(receipt,result)
    bounds=[receipt,BASE/'run_fit_durable.ps1',auditpath,LABEL/'compatibility/report.json',
        LABEL/'labels/labels.npz',PRIOR/'fit/student_head.pt']
    draft=dict(kind='phase_only_training_clearance_draft',model_fitting_authorized=False,
        additional_updates=5000,ordinary_final_global_step=65000,full_batch_rows=3057,
        frozen_sources_sha256=sha(receipt),compatibility_report_sha256=sha(LABEL/'compatibility/report.json'),
        root_data_audit_sha256=sha(auditpath),source_review_pending=True,
        bound_files=[dict(path=str(path),sha256=sha(path)) for path in bounds],
        root_authorized_scope='ONE fixed5000 full-batch nine-cell cosine fit after frozen source and launcher review; ordinaryfinal65000 only. Then ONE fresh canonical BFM250 plus learned1019 plus terminal300 and continuous250 hold after final numerical export review.',
        hardware_authorized=False)
    write(BASE/'training_clearance_DRAFT.json',draft)
    write(BASE/'frozen_preparation_status.json',dict(frozen_receipt_sha256=sha(receipt),sources=len(sources),inputs=len(inputs),
        trainer_sha256=sources['fit_phase_fullbatch_once.py'],runtime_sha256=sources['evaluate_phase_student.py'],
        fit_launcher_sha256=sha(BASE/'run_fit_durable.ps1'),canonical_launcher_sha256=sha(BASE/'run_canonical_durable.ps1'),
        no_execution_started=True))
    print(json.dumps(read(BASE/'frozen_preparation_status.json')))

if __name__=='__main__':main()
