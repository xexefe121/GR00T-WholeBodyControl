"""Record completed source/launcher checks before constructing any fit packet."""
import ast, hashlib, json
import xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
names=('prepare_execution.py','execution_common.py','run_fit_durable_v1.ps1','dispatch_fit.ps1',
       'verify_completed.py','read_fit_progress_v1.ps1','test_execution_helpers.py','test_launcher.ps1','prove_actual_metadata_gate.py')
for name in names:
    if name.endswith('.py'):ast.parse((BASE/name).read_text(encoding='utf-8-sig'))
root=ET.parse(BASE/'root_execution_tests_v1.xml').getroot()
suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites)==37
assert all(int(s.attrib.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
ps=read(BASE/'launcher_tests.json');assert ps['passed'] is True and ps['count']==14 and all(ps['checks'].values())
assert ps['actual_task_dispatches']==ps['model_calls']==ps['native_steps']==0
proof=read(BASE/'metadata_schema_proof.json');assert proof['passed'] is True and proof['check_count']==81
source=read(BASE/'source_preparation.json');review=read(BASE/'root_review.json')
assert review['source_review_pass'] is True and review['source_sha256']==source['source_sha256']
assert sha(BASE/'selected_protocol.json')=='1cade1a88df2324511ca2b9599b69261e9a9b7a72ba34c540b09b8ac10896128'
for name,digest in source['source_sha256'].items():assert sha(Path(source['source_directory'])/name)==digest
assert not any((BASE/name).exists() for name in ('fit','training_request.json','training_clearance.json','fit_process_v1'))
evidence=('root_execution_tests_v1.xml','launcher_tests.json','metadata_schema_proof.json','metadata_gate_proof.json',
          'source_preparation.json','root_review.json','selected_protocol.json','launcher_derivation.patch')
result=dict(source_preparation_pass=True,source_review_pass=True,reviewer='root',helper_sha256={n:sha(BASE/n) for n in names},
    evidence_sha256={n:sha(BASE/n) for n in evidence},synthetic_tests=37,PS51_checks=14,metadata_lineage_checks=81,
    original_trainer_sources_unchanged=39,source_preparation_sha256=sha(BASE/'source_preparation.json'),
    scope=['Configuration-only metadata proof precedes streaming actual file rehash.',
      'Native512 source restored without expansion; old corpus/schedule/context filenames checked against actual manifests.',
      'Captured handles, exact receipt/clearance arguments, unknown-exit rejection and no automatic retry.',
      'Owner checks all request subjects, process arguments, source/runtime/input pins, dispatch and independently observed absence.',
      'Owner accounting does not establish model authenticity or physical qualification.'],
    task_array_loads=0,checkpoint_loads=0,model_calls=0,native_steps=0,actual_dispatch=False,writer_sha256=sha(__file__))
write(BASE/'execution_helper_preparation.json',result)
print(json.dumps({'passed':True,'helpers':len(names),'sha256':sha(BASE/'execution_helper_preparation.json')}))
