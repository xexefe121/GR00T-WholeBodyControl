"""Bind final independent CLEAR to the already selected single experiment."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
assert not any((BASE/name).exists() for name in ('runtime_clearance.json','canonical_launch_receipt.json','canonical_process_status.json','nominal','post_lifecycle_hold_5s'))
reviewpath=BASE.parent/'filtered_student_prelaunch_review_v1/review.json'
assert sha(reviewpath)=='9199ff06ed155076f33ce1765e7422b6333e52e3fb0e4200d64cb51352d2c8d1'
review=read(reviewpath);assert review['clear'] and not review['findings'] and all(review['checks'].values())
receipt=read(BASE/'frozen_inputs_v2.json')
assert sha(BASE/'frozen_inputs_v2.json')==review['frozen_receipt_sha256']=='fe3edfdecb5748c2f58a10c2dc760b3db0a8bfc3a55d470bf73d58b59186f6f6'
for name,digest in receipt['source_sha256'].items():assert sha(BASE/'source_snapshot_v1'/name)==digest
for path,digest in receipt['input_sha256'].items():assert sha(Path(path))==digest
clearance=read(BASE/'runtime_clearance_DRAFT.json')
assert not clearance['canonical_rollout_authorized'] and not clearance['root_selection_pending'] and clearance['source_review_pending']
for item in clearance['bound_files']:assert sha(Path(item['path']))==item['sha256']
assert read(BASE/'root_selection.json')['selected'] and clearance['canonical_trials']==1
clearance.update(kind='reviewed_root_selected_filtered_runtime_clearance',canonical_rollout_authorized=True,
    source_review_pending=False,review_sha256=sha(reviewpath),
    draft_sha256=sha(BASE/'runtime_clearance_DRAFT.json'),ordinary_final_global_step=65000,
    independent_simulation_qualification_pending=True)
clearance['bound_files'].extend(dict(path=str(path),sha256=sha(path)) for path in (reviewpath,BASE/'runtime_clearance_DRAFT.json'))
dest=BASE/'runtime_clearance.json';dest.write_text(json.dumps(clearance,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(clearance_sha256=sha(dest),review_sha256=sha(reviewpath),canonical_trials=1)))
