"""Copy qualified pure modules only; no task arrays are opened."""
from pathlib import Path
import hashlib
import json
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
source=BASE/'source_draft_v1'
source.mkdir(exist_ok=True)
COPIES={
 'direct_features.py':NEW/'direct_target_causal_width512_evaluation_v1/source_draft_v1/direct_features.py',
 'causal_features.py':NEW/'direct_target_causal_width512_evaluation_v1/source_draft_v1/causal_features.py',
 'observations.py':NEW/'direct_target_causal_width512_evaluation_v1/source_draft_v1/gear_sonic/utils/g1_true23_bfm_seed_observations.py',
 'input_schema.py':NEW/'direct_target_full_state_secants_v1/source_snapshot_v1/input_schema.py',
}
rows={}
for name,path in COPIES.items():
    data=path.read_bytes()
    with (source/name).open('xb') as f:f.write(data)
    rows[name]={'path':path.as_posix(),'sha256':hashlib.sha256(data).hexdigest()}
with (BASE/'copied_sources.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps({'copied':len(rows),'task_arrays_opened':0}))
