"""Freeze narrow split source preparation; no task data/model execution."""
from pathlib import Path
import hashlib,json,shutil
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_causal_context_study_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def item(path):return dict(path=Path(path).as_posix(),sha256=sha(path))
def main():
    source=BASE/'source_draft_v1';target=BASE/'source_prepared_v1';target.mkdir(exist_ok=False)
    hashes={p.name:sha(p) for p in sorted(source.glob('*.py'))}
    old=read(OLD/'training_frozen_inputs.json')['source_sha256']
    for name,digest in hashes.items():
        shutil.copyfile(source/name,target/name)
        if sha(target/name)!=digest:raise ValueError('Source copy differs.')
    unchanged=[name for name in hashes if old.get(name)==hashes[name]]
    if len(unchanged)!=17 or set(hashes)-set(old)!={'test_split_first_layer.py'}:raise ValueError('Unexpected source scope.')
    subjects={name:item(BASE/name) for name in ('DESIGN.md','OUTPUT_SCHEMA.md','training_request_proposal.json','source_derivation.json','prepare_split.py','record_preparation.py','synthetic_tests_v1.xml','export_tests_v1.xml','test_context_export.py')}
    subjects.update(original_source_preparation=item(OLD/'source_preparation.json'),
        completed_context_proof=item(OLD/'context_preflight/report.json'),
        completed_context_source_review=item(BASE.parent/'direct_target_context_source_review_v1/review.json'),
        failed_attempt_owner=item(OLD/'owner_completion_verification_v3.json'),
        failed_attempt_root_audit=item(BASE.parent/'direct_target_context_initial_failure_root_v1/report.json'))
    result=dict(source_preparation_passed=True,preparation_only=True,source_directory=target.as_posix(),source_sha256=hashes,
        unchanged_original_modules=unchanged,changed_original_modules=['context_model.py','train_context_pair.py'],
        new_synthetic_test='test_split_first_layer.py',subjects=subjects,synthetic_tests_passed=25,
        context_data_proof_reused_without_rerun=True,objective_data_schedule_normalization_unchanged=True,
        initial_actual_gate_rad=1e-5,final_actual_gate_rad=1e-5,split_Torch32_monolithic_export64=True,
        prior_task_initial_forwards=1437,prior_task_initial_rows=367570,prior_task_optimizer_updates=0,
        actual_task_model_calls=0,actual_gradient_calls=0,actual_optimizer_updates=0,native_steps=0,
        actual_training_request_present=(BASE/'training_request.json').exists(),actual_training_clearance_present=(BASE/'training_clearance.json').exists())
    path=BASE/'source_preparation.json'
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(source_preparation_sha256=sha(path),sources=len(hashes),synthetic_tests=25)))
if __name__=='__main__':main()
