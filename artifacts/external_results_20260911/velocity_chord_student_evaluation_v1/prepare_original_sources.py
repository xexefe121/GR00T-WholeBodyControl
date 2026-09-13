"""Preserve original unfiltered evaluator/runtime source; preparation only."""
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).parent
OLD=BASE.parent/'fast_controller_phase_fit_v1/source_snapshot_v1'
DEST=BASE/'source_snapshot_v1'
DEST.mkdir(exist_ok=True)
copied={}
for src in OLD.rglob('*.py'):
    relative=src.relative_to(OLD)
    dst=DEST/relative;dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert dst.read_bytes()==src.read_bytes()
    else:shutil.copyfile(src,dst)
    copied[str(relative).replace('\\','/')]=dict(original=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest())
for name in ('gear_sonic/__init__.py','gear_sonic/utils/__init__.py'):
    p=DEST/name
    if not p.exists():p.write_text('')
    assert not p.read_bytes()
(BASE/'original_sources.json').write_text(json.dumps(copied,indent=2)+'\n')
print(json.dumps(dict(original_python_files=len(copied),physics_steps=0,inference_calls=0)))
