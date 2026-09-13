"""Filesystem-only import snapshot; no simulator or policy execution."""
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).parent
OLD = OUT.parent / 'preserved_walk_demo_v1'
for item in json.loads((OLD/'mapping.json').read_text()):
    source = Path(item['frozen'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    target = OUT/'repo'/source.relative_to(OLD/'repo')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): assert target.read_bytes() == source.read_bytes()
    else: shutil.copyfile(source, target)
hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'repo').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
(OUT/'import_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(len(hashes))
