"""Independent static export review; no model imports, forwards or task arrays."""
import ast,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'direct_target_width512_export_preparation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep_path=BASE/'source_preparation.json'
assert sha(prep_path)=='5f0dd117cec50f6501f9ed6cf44bc687740b2ef007dfb2f462420a4072b3f2ee'
prep=read(prep_path)
assert prep['source_preparation_passed'] is True
source=Path(prep['source_directory'])
for n,h in prep['source_sha256'].items():assert sha(source/n)==h,n
for p,h in prep['input_sha256'].items():assert sha(p)==h,p
old=NEW/'direct_target_causal_response_balanced_student_v2/source_snapshot_v1/response_promoted.py'
text=old.read_text(encoding='utf-8-sig')
for before,after in prep['source_changes']:
    assert text.count(before)==1,(before,text.count(before))
    text=text.replace(before,after)
assert text==(source/'width512_promoted.py').read_text(encoding='utf-8-sig')
tree=ast.parse(text)
nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
oldtree=ast.parse(old.read_text(encoding='utf-8-sig'))
oldnodes={n.name:n for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert ast.dump(nodes['PromotedDirect'],include_attributes=False)==ast.dump(oldnodes['PromotedDirect'],include_attributes=False)
export=nodes['export_onnx']
calls=[ast.unparse(n.func) for n in ast.walk(export) if isinstance(n,ast.Call)]
assert all(n not in calls for n in ('model','model.forward','F.linear','torch.onnx.export','torch.load'))
assert prep['synthetic_tests']==20 and prep['tests_failed']==prep['tests_skipped']==0
xml=ET.parse(BASE/'synthetic_tests.xml').getroot()
suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites)==20
assert all(int(s.attrib.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
assert prep['final_admission_backends']==['CPU64','GPU64','ORT64']
assert prep['unchanged_parity_tolerance_rad']==1e-5 and prep['final_GPU32_drift_is_diagnostic'] is True
assert prep['initialization_GPU32_comparison_gate_preserved'] is True
assert all(prep[k]==0 for k in ('actual_checkpoint_loads','task_arrays_loaded','Torch_forward_calls','ORT_sessions','ORT_calls','native_steps','optimizer_updates'))
assert prep['cuda_initialized'] is False
result=dict(passed=True,source_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_subject={'path':prep_path.as_posix(),'sha256':sha(prep_path)},source_sha256=prep['source_sha256'],
 four_literal_changes_only=True,original_promoted_class_AST_exact=True,producer_synthetic_tests_verified=20,
 root_test_reruns=0,all_preparation_pins_exact=True,writer_sha256=sha(__file__),
 reviewed_semantics=['Same manual20-node FP64 graph with float32 public1323/23 tensors and exact promoted original float32 parameters.',
 'Only ordinary81000 endpoint,512 shapes, error text and graph label changed; original math and buffer ownership unchanged.',
 'Final CPU64/GPU64/ORT64 original1e-5rad gate; finalGPU32 drift separately diagnostic. Initial GPU32 source comparison gate unchanged.',
 'Exporter alone does not qualify causal lineage, optimizer, full-data fit, runtime latency or physical behavior; actual fit release remains required.'],
 actual_checkpoint_loads=0,task_arrays_loaded=0,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,
 actual_fit_selected=False,execution_clearance=False,behavioral_qualification=False)
path=OUT/'review.json'
with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'path':path.as_posix(),'sha256':sha(path),'passed':True,'producer_tests_verified':20}))
