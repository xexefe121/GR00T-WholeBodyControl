import ast,hashlib,json
from pathlib import Path
B=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_fp64_root_audit_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prep=B/'source_preparation_v2.json';assert sha(prep)=='b32c89a04601c361fb68db7c116c1f510b5b9b7c221278fc5e895d40d19c157f'
j=json.loads(prep.read_text());pins=dict(j['source_sha256'])
for path,digest in pins.items():assert sha(path)==digest;ast.parse(Path(path).read_text())
for name in ('audit_graph.py','audit_arrays.py','audit_math.py'):assert sha(B/name)==sha(B/'source_v2'/name)
old=(B/'audit_saved_export.py').read_text();new=(B/'source_v2/audit_saved_export.py').read_text()
before="export_manifest=sha(export/'manifest.json'),root_training_audit="
after="export_manifest=sha(frozen_path),output_manifest=sha(export/'manifest.json'),root_training_audit="
assert old.count(before)==1 and old.replace(before,after)==new
assert sha(B/'graph_tests.xml')=='64695ecee3a4fc49b3279d1ffffe8722850c870a4851b34b73652875d4918753'
pins[str(prep)]=sha(prep);pins[str(B/'graph_tests.xml')]=sha(B/'graph_tests.xml')
report={'kind':'independent_saved_fp64_export_audit_source_review','passed':True,'source_review_pass':True,
 'source_sha256':pins,'actual_audit_executed_by_reviewer':False,'findings':[],
 'checks':['Verified complete20-node graph wiring/attributes/opset/dtypes and exact original f32 checkpoint weights/normalization promoted64, including transposed MatMul constants and bounded double ELU decomposition.',
 'All9 preclamp comparisons,27 full drift arrays with six clipping-mask counts and1803 exact call partitions/counters independently reconstructed from saved outputs. Same default/span widening and unchanged1e-5 tolerance.',
 'Independent metrics retain correct5x3 nominal order, full velocity/physical response groups and float32 nominal reduction tolerance; audit_math byte-identical to previously reviewed source6313988b.',
 'Actual request/source/input/clearance/manifest/report/failed-training identities bind before final source/input rehash. Owner/process completion is a separate required final review subject.',
 'Fresh v2 changes only export_manifest release role to export_frozen_inputs.json and retains separate output_manifest identity. V1 source preserved. Five prior graph-corruption tests apply unchanged.',
 'No network, model inference, ORT evaluation, optimizer or native steps. Saved checkpoint deserialization and ONNX tensor/graph inspection only.'],
 'scope':'One completed-export saved evidence audit after producer completion. Its pass cannot qualify controller balance, real-time timing or hardware.',
 'reviewer_model_calls':0,'reviewer_native_steps':0,'reviewer_optimizer_updates':0}
out=Path(__file__).parent/'review.json'
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(sha(out))
