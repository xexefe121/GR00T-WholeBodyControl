"""Read-only source binding for the narrow opt-in K=0 restoration retry."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPO = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
OLD = HERE.parent/'restoration_stall_3805_v1'
OUT = HERE/'retry_review'
OUT.mkdir(exist_ok=True)
files = [REPO/'gear_sonic/utils/g1_true23_mjbatch_restoration.py',
         REPO/'gear_sonic/scripts/continue_g1_true23_mpc_hard_feasibility.py',
         REPO/'gear_sonic/tests/test_g1_true23_mjbatch_feasibility.py',
         REPO/'gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py',
         REPO/'gear_sonic/utils/g1_true23_mjbatch_mpc.py']
inventory=[]
for path in files:
    content=path.read_bytes()
    (OUT/path.name).write_bytes(content)
    inventory.append(dict(path=str(path),sha256=hashlib.sha256(content).hexdigest()))

def methods(path):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Native23RestorationTracker')
    return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}

before,after=methods(OLD/'g1_true23_mjbatch_restoration.py'),methods(files[0])
unchanged={name:ast.dump(before[name],include_attributes=False)==ast.dump(after[name],include_attributes=False)
           for name in ('residual','target_reference')}
assert all(unchanged.values())
ctor=after['__init__']
defaults=dict(zip([a.arg for a in ctor.args.kwonlyargs],ctor.args.kw_defaults))
assert isinstance(defaults['zero_rollout_feedback'],ast.Constant) and defaults['zero_rollout_feedback'].value is False
main_unchanged={p.name:p.read_bytes()==(OLD/p.name).read_bytes() for p in files[-2:]}
assert all(main_unchanged.values())
summary=dict(source_hashes=inventory,restoration_residual_and_target_reference_AST_unchanged=unchanged,
             default_zero_rollout_feedback_is_false=True,main_core_and_tracker_bytes_unchanged=main_unchanged,
             tests_or_physics_executed_by_reviewer=False)
(OUT/'source_binding.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2))
