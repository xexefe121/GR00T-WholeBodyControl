"""Record source-only design provenance; no checkpoint/data/model imports."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SOURCE=NEW/'direct_target_causal_width512_student_v1/source_prepared_v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
names=['train_response_balanced.py','response_contract.py','response_data.py','response_restore.py','width_restore.py',
       'width512.py','width512_promoted.py','direct_contract.py','direct_objective.py','full_state_data.py',
       'full_state_objective.py','balanced_response_objective.py','context_diagnostics.py']
refs=[SOURCE/name for name in names]+[
    NEW/'direct_target_width251_followup_design_v1/DESIGN.md',
    NEW/'direct_target_width251_collection_v1/source_preparation.json',
    NEW/'direct_target_width251_collection_v1/OUTPUT_SCHEMA.md',
    NEW/'direct_target_width251_collection_root_review_v1/review.json']
result=dict(source_only=True,design_complete=True,implementation_selected=False,training_selected=False,
    update_count=None,learning_rate=None,new_supervision_weight=None,new_sampler=None,
    unchanged_benchmarks=dict(nominal_rows=9904,nominal_cells=15,physical_rows=3054,physical_cells=9,
        full_state_rows=354612,full_state_cells=54,diagnostic_rows=367570),
    conditional_new_data=dict(rows=1018,controls=[251,1268],phase_rows=[99,819,100],fresh_student_state_queries=1,
        connected_expert_rows_after_first=1017),
    restoration=dict(source_ordinary_step=81000,source_optimizer_step=16000,architecture=[1323,512,512,23],
        all_six_weights_and_moments_preserved=True,expansion_repeated=False,normalization_refit=False),
    conditional_full_branch_accounting=dict(training_calls_per_update=4,training_rows_per_update=15704,
        diagnostic_rows_per_backend=368588,diagnostic_calls_per_backend=1441,
        note='Conditional formulas only. No actual numerical budget selected.'),
    task_checkpoints_opened=0,task_arrays_opened=0,model_calls=0,gradient_calls=0,native_steps=0,replans=0,
    references={p.as_posix():sha(p) for p in refs},design_sha256=sha(BASE/'DESIGN.md'),source_sha256=sha(Path(__file__)))
with (BASE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(report=str(BASE/'report.json'),sha256=sha(BASE/'report.json'))))
