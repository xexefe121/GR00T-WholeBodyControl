import ast,hashlib,json,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'independent_plant_pending_publication_v1';SOURCE=BASE/'source_draft_v1'
OLD=NEW/'independent_plant_clock_timeout_correction_v1/source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(BASE/'source_preparation.json')=='6cb9f8bf38543634c32e745496f63fe94e23391fd10948be41749eaea5356575'
p=read(BASE/'source_preparation.json')
assert p['preparation_passed'] is True and len(p['source_sha256'])==22
for name,digest in p['source_sha256'].items():assert sha(SOURCE/name)==digest
for name,digest in p['original_source_sha256'].items():assert sha(OLD/name)==digest
assert p['changed_original_files']==['clock_core.py'] and len(p['unchanged_original_files'])==20
for name in p['unchanged_original_files']:assert sha(SOURCE/name)==sha(OLD/name)
for path,digest in p['input_sha256'].items():assert sha(path)==digest,path
def methods(path):
    cls=next(v for v in ast.parse(path.read_text()).body if isinstance(v,ast.ClassDef) and v.name=='PlantFoundation')
    return {v.name:ast.dump(v,include_attributes=False) for v in cls.body if isinstance(v,ast.FunctionDef)}
old,new=methods(OLD/'clock_core.py'),methods(SOURCE/'clock_core.py')
for name in p['unchanged_foundation_method_AST']:assert new[name]==old[name],name
run=subprocess.run([sys.executable,'-m','pytest','--rootdir=.','--confcutdir=.','-q','--junitxml='+str(OUT/'root_tests.xml')],cwd=SOURCE,capture_output=True,text=True)
(OUT/'root_tests.log').write_text(run.stdout+'\n'+run.stderr)
assert run.returncode==0,run.stdout+run.stderr
xml=ET.parse(OUT/'root_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
assert sum(int(s.attrib['tests']) for s in suites)==72
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
report=dict(passed=True,source_review_pass=True,source_preparation_sha256=sha(BASE/'source_preparation.json'),
 source_sha256=p['source_sha256'],input_sha256=p['input_sha256'],reviewer='root',unchanged_original_sources=20,
 unchanged_foundation_method_AST=p['unchanged_foundation_method_AST'],root_synthetic_tests=67,root_subtests=5,
 reviewed_semantics=['one immutable original Job and payload; history and snapshot unchanged across retry',
 'only returned BUSY is retryable; at most initial plus9 attempts, once per native tick before original activation',
 'no republish after PUBLISHED, no retry after exception or unknown transport outcome',
 'retry guard timestamp and actual transport times separate; scheduler stalls remain observable; original admission rejects late results',
 'expiration preserves existing command-deadline fault and held-command behavior',
 'attempted/returned/logged counters and last immutable evidence distinguish transport and logging failures',
 'original 2ms/20ms/debt/elapsed/native caps, worker, model, strict oracle and watchdog unchanged',
 'bounded records fit original80000 capacity; no actual timing qualification inferred from fakes'],
 saved_retry_protocol_audit_review_required=True,actual_run_selected=False,actual_binding_created=False,
 model_calls=0,ORT_calls=0,gradient_calls=0,native_steps=0,spawned_workers=0,task_arrays_loaded=0,
 root_test_log_sha256=sha(OUT/'root_tests.log'),writer_sha256=sha(__file__))
path=OUT/'review.json'
with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(path),tests=67,subtests=5)))
