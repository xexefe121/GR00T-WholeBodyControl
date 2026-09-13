"""Bind final independent CLEAR to the already selected single experiment."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
assert not any((BASE/name).exists() for name in ('runtime_clearance.json','canonical_launch_receipt.json','canonical_process_status.json','nominal','post_lifecycle_hold_5s'))
reviewpath=BASE.parent/'cost_ranked_student_prelaunch_review_v1/review.json'
assert sha(reviewpath)=='ad35732e0dc06430780f832371f6852c94bce879aab7af4c7890818478fef9a3'
review=read(reviewpath)
assert review['verdict']=='CLEAR' and review['root_selected_canonical_trials']==1
assert review['source_hashes_verified']==26 and review['input_hashes_verified']==174 and review['powershell_ast_errors']==0
assert not review['existing_output_or_process_receipts_at_review']
for name,digest in review['bindings'].items():assert sha(BASE/name)==digest
receipt=read(BASE/'frozen_inputs_v2.json')
assert sha(BASE/'frozen_inputs_v2.json')==review['bindings']['frozen_inputs_v2.json']=='9343c704c68678fd4478fba104e2410631dd2cc785aeb278e476257e5c06fa49'
for name,digest in receipt['source_sha256'].items():assert sha(BASE/'source_snapshot_v1'/name)==digest
for path,digest in receipt['input_sha256'].items():assert sha(Path(path))==digest
clearance=read(BASE/'runtime_clearance_DRAFT.json')
assert not clearance['canonical_rollout_authorized'] and not clearance['root_selection_pending'] and clearance['source_review_pending']
for item in clearance['bound_files']:assert sha(Path(item['path']))==item['sha256']
assert read(BASE/'root_selection.json')['selected'] and clearance['canonical_trials']==1
clearance.update(kind='reviewed_root_selected_cost_ranked_runtime_clearance',canonical_rollout_authorized=True,
    source_review_pending=False,review_sha256=sha(reviewpath),
    draft_sha256=sha(BASE/'runtime_clearance_DRAFT.json'),ordinary_final_global_step=65000,
    independent_simulation_qualification_pending=True)
clearance['bound_files'].extend(dict(path=str(path),sha256=sha(path)) for path in (reviewpath,BASE/'runtime_clearance_DRAFT.json'))
dest=BASE/'runtime_clearance.json';dest.write_text(json.dumps(clearance,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(clearance_sha256=sha(dest),review_sha256=sha(reviewpath),canonical_trials=1)))
