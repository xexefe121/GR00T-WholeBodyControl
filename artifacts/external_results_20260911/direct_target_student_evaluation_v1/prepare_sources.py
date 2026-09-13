"""Copy immutable evaluator dependencies and derive only direct feature math."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SOURCE=BASE/'source_draft_v1'
SOURCE.mkdir(exist_ok=False)
originals=json.loads((NEW/'velocity_chord_student_evaluation_v1/original_sources.json').read_text())
pins={}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name,item in originals.items():
    original=Path(item['original']);assert sha(original)==item['sha256']
    dest=SOURCE/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,dest)
    pins[name]=dict(original=str(original),sha256=sha(dest))
prior=NEW/'one_step_physical_student_evaluation_v1/source_snapshot_v1'
for name,dest in [('evaluate_physical_response_student.py','original_evaluator.py'),
                  ('proposal_evidence.py','original_proposal_evidence.py'),
                  ('evaluation_gate.py','original_evaluation_gate.py'),
                  ('head_activation_witness.py','original_head_activation_witness.py')]:
    shutil.copy2(prior/name,SOURCE/dest);pins[dest]=dict(original=str(prior/name),sha256=sha(SOURCE/dest))
text=(SOURCE/'gear_sonic/utils/g1_true23_mpc_student.py').read_text()
node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='GoalFeatures')
feature=ast.get_source_segment(text,node)
edits=[('class GoalFeatures:', 'class DirectFeatures:'),
       ('def __call__(self,qpos,qvel,previous_target,frame):','def __call__(self,qpos,qvel,frame):'),
       ('np.asarray(previous_target)-self.default,np.asarray(qvel[:3])@heading,qpos[2]',
        'np.asarray(qvel[:3])@heading,qpos[2]'),
       ('proprio.shape==(79,)','proprio.shape==(56,)')]
for old,new in edits:
    assert feature.count(old)==1,old
    feature=feature.replace(old,new)
(SOURCE/'direct_features.py').write_text('"""Pure 1000-feature state/received-goal builder. No BFM or history inputs."""\nimport numpy as np\nfrom scipy.spatial.transform import Rotation\nOFFSETS=np.array([0,1,2,4,8,16,24,37],dtype=np.int64)\nFEATURES=1000\n\n'+feature+'\n')
# Path/hash/JSON helpers are pure; preserve their exact source segments.
text=(SOURCE/'student_linear_runtime.py').read_text()
nodes=ast.parse(text).body
names={'sha','archive','finite_json'}
helpers='\n\n'.join(ast.get_source_segment(text,n) for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names)
(SOURCE/'runtime_common.py').write_text('"""Pure artifact path and serialization helpers."""\nfrom pathlib import Path\nimport hashlib,json,sys\nimport numpy as np\nBASE=Path(__file__).resolve().parent.parent\nif sys.platform=="win32":\n    ROOT=Path("Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")\n    TASK=Path("E:/codex-artifacts/sonic23_teleop_six_hour_20260910")\n    DEPS=None\nelse:\n    ROOT=Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")\n    TASK=Path("/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910")\n    DEPS=Path("/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps")\nBUNDLE=ROOT/"artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"\nONNX=ROOT/"artifacts/teleop_six_hour_20260910/bfm_onnx_v2"\nREFERENCE=TASK/"mjbatch_intent_floor_inputs_v1/walk003/reference.npz"\nTEACHER=TASK/"bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1"\nKIND="native23_direct_absolute_target_student_v1"\nFEATURES=1000\n\n'+helpers+'\n')
with (BASE/'original_sources.json').open('x') as stream:json.dump(pins,stream,indent=2)
with (BASE/'feature_derivation.json').open('x') as stream:json.dump(dict(
    original_sha256=pins['gear_sonic/utils/g1_true23_mpc_student.py']['sha256'],
    direct_sha256=sha(SOURCE/'direct_features.py'),edits=edits,
    retained_slices=[[0,52],[75,1023]],feature_count=1000,model_calls=0,native_steps=0),stream,indent=2)
print('Prepared',len(pins),'byte-preserved dependencies; pure direct features derived.')
