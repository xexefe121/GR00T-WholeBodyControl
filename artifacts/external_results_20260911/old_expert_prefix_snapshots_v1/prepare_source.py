"""Prepare a private capture source; no physics or inference."""
import ast
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).parent;SOURCE=BASE/'source_snapshot_v1';SOURCE.mkdir(exist_ok=True)
OLD=BASE.parent/'fast_controller_phase_fit_v1/source_snapshot_v1'
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
copied={}
for path in OLD.rglob('*.py'):
    target=SOURCE/path.relative_to(OLD);target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(path.read_bytes())
    copied[path.relative_to(OLD).as_posix()]=sha(target)
for path in (SOURCE/'gear_sonic/__init__.py',SOURCE/'gear_sonic/utils/__init__.py'):path.write_text('')
upstream=ROOT/'artifacts/teleop_resume_20260911/reconstruct_control_snapshots.py'
(BASE/'upstream_reconstruct_control_snapshots.py').write_bytes(upstream.read_bytes())
oracle_source=BASE.parent/'velocity_chord_student_evaluation_v1/source_snapshot_v1/evaluate_velocity_chord_student.py'
node=next(n for n in ast.parse(oracle_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='assess')
(SOURCE/'strict_native.py').write_text('"""Unchanged selected native2ms oracle, AST extracted without evaluator import."""\nimport numpy as np\n\n'+ast.unparse(node)+'\n')
new=next(n for n in ast.parse((SOURCE/'strict_native.py').read_text()).body if isinstance(n,ast.FunctionDef))
assert ast.dump(node)==ast.dump(new)
receipt=dict(original_sources=copied,original_source_count=len(copied),upstream_capture_sha256=sha(upstream),
    oracle_source_sha256=sha(oracle_source),strict_oracle_sha256=sha(SOURCE/'strict_native.py'),strict_oracle_ast_unchanged=True,
    inference_calls=0,physics_steps=0)
(BASE/'source_derivation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(original_sources=len(copied),strict_oracle_ast_unchanged=True)))
