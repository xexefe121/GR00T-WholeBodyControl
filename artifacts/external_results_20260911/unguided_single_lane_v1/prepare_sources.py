"""Copy immutable current-run snapshots into a private import tree; no shared edits."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
RUN=HERE.parent/'pico_full_hard_restoration_v1'
DEST=HERE/'source_snapshot/gear_sonic/utils'
DEST.mkdir(parents=True,exist_ok=False)
mapping={}
for source in RUN.glob('*_snapshot.py'):
    if source.name.startswith('evaluate_'):
        continue
    target=DEST/source.name.replace('_snapshot.py','.py')
    shutil.copyfile(source,target)
    mapping[str(source)]=dict(copy=str(target),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(HERE/'source_binding.json').write_text(json.dumps(mapping,indent=2)+'\n')
print(json.dumps({Path(k).name:v['sha256'] for k,v in mapping.items()}))
