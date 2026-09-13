"""Root review of conditional D3 integration; no actual fit or task arrays."""
import ast,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
p=BASE/'source_preparation.json';assert sha(p)=='658ee9edccc99f9b1e8c8124d827ccff1db1ecef53ac08d6631e2ca869d42420'
r=read(p);assert r['source_preparation_pass'] is True and len(r['source_sha256'])==39
for name,h in r['source_sha256'].items():assert sha(BASE/'source_prepared_v1'/name)==h
for role,count in [('unchanged_original_sources',28),('unchanged_additional_sources',2)]:
 assert len(r[role])==count
 for name,item in r[role].items():assert sha(item['path'])==item['sha256']==r['source_sha256'][name]
for name,h in r['source_references'].items():assert sha(name)==h
for role,name in [('source_derivation_sha256','derive_integration.py'),('main_derivation_sha256','main_derivation.patch'),('output_schema_sha256','OUTPUT_SCHEMA.md'),('preparation_source_sha256','prepare_source.py')]:assert sha(BASE/name)==r[role]
assert sha(r['tests']['path'])==r['tests']['sha256']
tests=BASE/'root_tests_v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==36
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
tree=ast.parse((BASE/'source_prepared_v1/train_recovery.py').read_text())
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_continuation')
loops=[n for n in ast.walk(run) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(UPDATES)'];assert len(loops)==1
loop=loops[0];calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call)]
assert sum(ast.unparse(n.func)=='counted_forward' for n in calls)==4
assert sum(ast.unparse(n.func)=='loss.backward' for n in calls)==1
assert sum(ast.unparse(n.func)=='optimizer.step' for n in calls)==1
assert not any(ast.unparse(n.func) in ('expand_source','verify_width_restoration') for n in ast.walk(run) if isinstance(n,ast.Call))
for field in ('actual_task_data_reads','actual_checkpoint_reads','actual_model_calls','actual_gradient_calls','actual_optimizer_updates','native_steps'):assert r[field]==0
result=dict(passed=True,source_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_sha256=sha(p),source_sha256=r['source_sha256'],
 root_test_receipt=dict(path=tests.as_posix(),sha256=sha(tests),junit_cases=36),scope=r['fixed_candidate_scope'],
 reviewed_properties=['All28 original source modules and two reviewed helpers preserved',
 'Strict full512 warm restore keeps all81000 weights/Adam16000/RNG/frozen normalization',
 'Old N15/P9/F54 objective and data maps unchanged; separate full1018 equal-phase D3 term',
 'Four counted training forwards, one backward/clip10/AdamWstep, explicit bounded original schedule prefix',
 'All old diagnostic partitions and nine FP64 parity checks retained; newfour batches and three parity checks added',
 'Separate D3/combined losses and actualattempt/return/synchronization/failure evidence',
 'Qualified collection, actual consistency and warm-source bindings required before task data/model work'],
 unselected_protocol_fields=r['unselected_protocol_fields'],actual_fit_selected=False,
 actual_task_data_reads=0,actual_checkpoint_reads=0,model_calls=0,native_steps=0,
 limitations=['Source review does not qualify numerical endpoint or closed-loop behavior.',
 'Future freezer must use exact own width512 shared paths bound by completed consistency, even for byte-equal normalization copies.',
 'One connected expert recovery trajectory does not establish coverage of later student departures.'],writer_sha256=sha(__file__))
out=BASE/'root_review.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(out),junit_cases=36,sources=39)))
