"""Derive one isolated yaw2->4 terminal variant from the frozen v1 runner."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import textwrap

BASE=Path(__file__).resolve().parent
CASE=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk003_terminal_bfm_hybrid_v1')
GOAL=BASE/'bfm_terminal_yaw4_goal.py'
RUNNER=BASE/'evaluate_walk003_terminal_bfm_yaw4_hybrid.py'
RECEIPT=BASE/'walk003_terminal_yaw4_derivation_receipt.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

frozen_helper=CASE/'source_g1_true23_mjbatch_bfm_seed.py'
helper_text=frozen_helper.read_text()
klass=next(n for n in ast.parse(helper_text).body if isinstance(n,ast.ClassDef) and n.name=='Native23BFMRolloutSeed')
method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_goal')
old=textwrap.dedent('\n'.join(helper_text.splitlines()[method.lineno-1:method.end_lineno]))+'\n'
assert old.count('np.clip(2*dy,-.8,.8)')==1
new=old.replace('def _goal(', 'def terminal_goal_yaw4(').replace('np.clip(2*dy,-.8,.8)','np.clip(4*dy,-.8,.8)')
old_ast=ast.parse(old)
old_ast.body[0].name='terminal_goal_yaw4'
class Transform(ast.NodeTransformer):
    changes=0
    def visit_BinOp(self,node):
        node=self.generic_visit(node)
        if isinstance(node.op,ast.Mult) and isinstance(node.left,ast.Constant) and node.left.value==2 and isinstance(node.right,ast.Name) and node.right.id=='dy':
            node.left.value=4;self.changes+=1
        return node
transform=Transform();expected=transform.visit(copy.deepcopy(old_ast))
assert transform.changes==1
assert ast.dump(expected,include_attributes=False)==ast.dump(ast.parse(new),include_attributes=False)
goal_text='"""Frozen BFM goal method; sole numerical change is terminal yaw gain2 to4."""\nimport numpy as np\nfrom gear_sonic.utils.g1_true23_bfm_seed_observations import _quaternion_matrix\n\n'+new
frozen_runner=CASE/'runner_snapshot.py'
runner=frozen_runner.read_text()
assert runner.count('import time\n')==1
runner=runner.replace('import time\n','import time\nfrom types import MethodType\n')
needle='    seed=Native23BFMRolloutSeed(native,contract,original,args.onnx,dependency_directory=args.dependencies,threads=1)\n'
assert runner.count(needle)==1
runner=runner.replace(needle,needle+'    from artifacts.teleop_six_hour_20260910.bfm_terminal_yaw4_goal import terminal_goal_yaw4\n    seed._goal=MethodType(terminal_goal_yaw4,seed)\n')
needle="    paths += [args.onnx/k for k in ('manifest.json','actor.onnx','backward.onnx')]\n"
assert runner.count(needle)==1
runner=runner.replace(needle,needle+"    paths += [args.repo/'artifacts/teleop_six_hour_20260910'/name for name in ('bfm_terminal_yaw4_goal.py','walk003_terminal_yaw4_derivation_receipt.json')]\n")
assert runner.count('BFM_identity=seed.identity(),motion_override=')==1
runner=runner.replace('BFM_identity=seed.identity(),motion_override=',
    "BFM_identity=dict(unmodified_base_helper=seed.identity(),effective_terminal_yaw_gain=4.,goal_override_sha256=sha(args.repo/'artifacts/teleop_six_hour_20260910/bfm_terminal_yaw4_goal.py')),motion_override=")
assert runner.count('BFM_position_gain=1.,BFM_yaw_gain=2.,BFM_goal_horizon=8,')==1
runner=runner.replace('BFM_position_gain=1.,BFM_yaw_gain=2.,BFM_goal_horizon=8,','BFM_position_gain=1.,BFM_yaw_gain=4.,BFM_goal_horizon=8,')
needle="            BFM_controls=int(np.sum(arrays['controller_mode']==1)),BFM_original_goal_reference=provenance['BFM_goal_reference'],\n"
assert runner.count(needle)==1
runner=runner.replace(needle,needle+'            BFM_terminal_yaw_gain=4.,\n')
compile(goal_text,str(GOAL),'exec');compile(runner,str(RUNNER),'exec')
assert not any(p.exists() for p in (GOAL,RUNNER,RECEIPT))
GOAL.write_text(goal_text);RUNNER.write_text(runner)
receipt=dict(kind='exact_one_terminal_goal_gain_variant_derivation',
    frozen_v1_runner=str(frozen_runner),frozen_v1_runner_sha256=sha(frozen_runner),
    frozen_base_goal_helper=str(frozen_helper),frozen_base_goal_helper_sha256=sha(frozen_helper),
    generated_goal=str(GOAL),generated_goal_sha256=sha(GOAL),generated_runner=str(RUNNER),generated_runner_sha256=sha(RUNNER),
    changed_goal_AST_numeric_nodes=transform.changes,
    numerical_change='omega_z=clip(2*yaw_error,-.8,.8) -> clip(4*yaw_error,-.8,.8)',
    goal_other_AST_operations_identical=True,position_gain=1.,horizon=8,
    original_source_prefix_unchanged=True,standing_switch_control=1269,original_lifecycle_controls=1569,
    separate_extension_controls=250,quiet_thresholds_unchanged=True,physics_unchanged=True,
    new_goal_is_bound_only_on_private_hybrid_instance=True,shared_helper_files_edited=False)
RECEIPT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
