"""Freeze reviewed .py sources only; no final artifacts, bindings or launchers."""
import ast,json,hashlib,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;DRAFT=BASE/'source_draft_v1';SOURCE=BASE/'source_snapshot_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=BASE.parent/'one_step_physical_evaluation_preparation_review_v1/review.json'
assert sha(review)=='3fef12f2357fadd76113ed5ee3eb256182e3fd162772ec75b32d933686366f73'
record=json.loads((BASE/'source_only_check.json').read_text());assert record['passed']
assert not SOURCE.exists()
for relative,digest in record['source_sha256'].items():
    p=DRAFT/relative;assert p.suffix=='.py' and sha(p)==digest
    ast.parse(p.read_text());dest=SOURCE/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha(dest)==digest
receipt=dict(source_preparation_only=True,source_directory=SOURCE.as_posix(),source_sha256=record['source_sha256'],
    review_path=review.as_posix(),review_sha256=sha(review),no_pycache_copied=True,head_bound=False,witness_bound=False,model_calls=0,native_steps=0)
(BASE/'source_freeze.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(sources=len(record['source_sha256']),no_pycache_copied=True,model_calls=0,native_steps=0)))
