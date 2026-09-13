"""Independent audit copies original reviewed helpers, never producer validators."""
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).parent
OLD=BASE.parent/'bfm_entry250_labels_v1/source_snapshot_v1'
DEST=BASE/'source'
DEST.mkdir(exist_ok=True)
names=['student_linear_runtime.py','terminal_yaw4_goal.py']
names += [str(p.relative_to(OLD)).replace('\\','/') for p in (OLD/'gear_sonic/utils').glob('*.py')]
copies={}
for name in names:
    src,dst=OLD/name,DEST/name
    dst.parent.mkdir(exist_ok=True,parents=True)
    if dst.exists():assert dst.read_bytes()==src.read_bytes()
    else:shutil.copyfile(src,dst)
    copies[name]=dict(original=str(src),sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
for name in ('gear_sonic/__init__.py','gear_sonic/utils/__init__.py'):
    p=DEST/name
    if not p.exists():p.write_text('')
    assert not p.read_bytes()
(BASE/'copied_original_sources.json').write_text(json.dumps(copies,indent=2)+'\n')
print('Copied nine original helper files for independent audit.')
