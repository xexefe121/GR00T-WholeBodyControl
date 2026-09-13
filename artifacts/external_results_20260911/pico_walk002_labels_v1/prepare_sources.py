"""Copy the previously qualified collection dependencies, without executing them."""
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).parent
OLD = BASE.parent / 'bfm_entry250_labels_v1/source_snapshot_v1'
DEST = BASE / 'source_snapshot_v1'
DEST.mkdir(exist_ok=True)
names = ['student_linear_runtime.py', 'terminal_yaw4_goal.py']
names += [str(p.relative_to(OLD)).replace('\\', '/') for p in (OLD / 'gear_sonic/utils').glob('*.py')]
receipt = {}
for name in sorted(names):
    src, dst = OLD / name, DEST / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert dst.read_bytes() == src.read_bytes(), name
    else:
        shutil.copyfile(src, dst)
    receipt[name] = dict(original_path=str(src), sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
for name in ('gear_sonic/__init__.py', 'gear_sonic/utils/__init__.py'):
    dst = DEST / name
    if not dst.exists():
        dst.write_text('')
    assert not dst.read_bytes()
(BASE / 'copied_dependencies.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(copied_dependencies=len(receipt), inference_calls=0, physics_steps=0)))
