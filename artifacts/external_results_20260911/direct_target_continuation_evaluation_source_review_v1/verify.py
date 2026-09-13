"""Source-only final55000 evaluator adaptation and explicit runtime inventory."""
import ast,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=NEW/'direct_target_continuation_evaluation_v1';OLD=NEW/'direct_target_student_evaluation_v1';OUT=Path(__file__).resolve().parent
pins={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def bind(p,d=None):
    p=Path(p).resolve();s=sha(p)
    if d is not None:assert s==d,(p,s,d)
    pins[p.as_posix()]=s;return p
def read(p,d=None):return json.loads(bind(p,d).read_text(encoding='utf-8-sig'))
def subject(p):p=bind(p);return dict(path=p.as_posix(),sha256=pins[p.as_posix()])
prep=read(BASE/'source_preparation.json','3b6321086160ab2f90281f465a71f2814a1a6f914216a92ef982eb57a784ce5a')
assert prep['passed'] is True and len(prep['source_sha256'])==30
derivation=read(BASE/'source_derivation.json','8075daf7080b14e94f1ba211e9c9e1ff82be8ba90bbb1c5766cf4598fdef495e')
assert len(derivation['exact_changes'])==2 and sum(v['byte_identical'] for v in derivation['sources'].values())==29
for name,digest in prep['source_sha256'].items():
    current=bind(BASE/'source_draft_v1'/name,digest);entry=derivation['sources'][name]
    prior=bind(entry['original'],entry['original_sha256']);assert digest==entry['sha256']
    text=prior.read_text()
    for change in derivation['exact_changes']:
        if change['file']==name:assert text.count(change['before'])==1;text=text.replace(change['before'],change['after'])
    assert current.read_text()==text,name
    assert ast.dump(ast.parse(current.read_text()),include_attributes=False)==ast.dump(ast.parse(text),include_attributes=False),name
for name,digest in prep['unchanged_helper_sha256'].items():
    bind(BASE/name,digest);bind(OLD/name,digest)
for path,digest in prep['evidence_sha256'].items():bind(path,digest)
count=0
for filename in ('source_stub_tests.xml','launch_helper_tests.xml'):
    suite=ET.parse(bind(BASE/filename)).getroot().find('testsuite')
    assert all(suite.attrib[k]=='0' for k in ('failures','errors','skipped'));count+=int(suite.attrib['tests'])
assert count==51
inventory=read(BASE/'runtime_inventory.json');assert inventory['recursive_training_hashes'] is False and inventory['total_files']==len(inventory['files'])==5184
assert sum(e['bytes'] for e in inventory['files'])==inventory['total_bytes']
oldpins=read(OLD/'evaluation_process/launch_receipt.json')['input_hashes'];overlap=0
for entry in inventory['files']:
    path=bind(entry['path'],entry['sha256']);assert path.stat().st_size==entry['bytes']
    if path.as_posix() in oldpins:assert oldpins[path.as_posix()]==entry['sha256'];overlap+=1
    assert 'direct_target_gpu_20260911' not in str(path) and '__pycache__' not in path.parts
for name in ('runtime_inventory.py','freeze_final_package.py','prepare_sources.py','prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py'):ast.parse(bind(BASE/name).read_text())
for name in ('witness_binding.json','evaluation_binding.json','head_witness','nominal','post_lifecycle_hold_5s'):assert not (BASE/name).exists(),name
bind(__file__)
report=dict(verdict='CLEAR',source_review_pass=True,preparation_only=True,
    subjects={name:subject(path) for name,path in [('source_preparation',BASE/'source_preparation.json'),('evaluator',BASE/'source_draft_v1/evaluate_direct_target_student.py'),('witness',BASE/'source_draft_v1/head_activation_witness.py'),('gate',BASE/'source_draft_v1/evaluation_gate.py'),('runtime_inventory',BASE/'runtime_inventory.json'),('freezer',BASE/'freeze_final_package.py'),('launcher',BASE/'prepare_bound_launcher.py')]},
    source_sha256=prep['source_sha256'],input_sha256=pins,unchanged_original_sources=29,total_runtime_sources=30,
    passed_existing_stub_tests=51,inventory_files=5184,inventory_bytes=inventory['total_bytes'],known_prior_launch_overlap_exact=overlap,
    changes='Only ordinary_final_step5000→55000 and additional_updates5000→50000 in runtime gate; full controller/strict oracle/history unchanged.',
    binding_review='Explicit actual runtime assets/packages plus small final dataset/fit/export/root/owner receipts; no recursive training/CUDA corpus traversal. Final dataset review must bind the actual training manifest.',
    limitations=['Package inventory is a conservative source-level closure, not a dynamic syscall trace. Established system CPython/OS libraries remain the trust boundary.',
        'Prior5000 behavioral trial failed; byte-preserved runtime arithmetic is not full-body qualification.',
        'Actual final55000 dataset/fit/export evidence, new one-call WSL witness and concrete launcher still require reviews before execution.'],
    execution_clearance=False,task_model_calls=0,native_steps=0,optimizer_updates=0,hardware_authorized=False)
with (OUT/'review.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'verdict':'CLEAR','review_sha256':sha(OUT/'review.json'),'inventory':5184,'prior_overlap_exact':overlap,'total_review_pins':len(pins)}))
