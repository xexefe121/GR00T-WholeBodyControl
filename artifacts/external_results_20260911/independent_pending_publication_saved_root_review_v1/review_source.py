"""Root review of retry-aware saved audit, including terminal evidence fixes."""
from pathlib import Path
import ast,hashlib,json,xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'independent_plant_pending_publication_saved_audit_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(BASE/'source_preparation_v2.json')=='e656ed762475a50f91cf60b4bfc3b59ce9c052ae44fad54c5474e41060adcfde'
p=read(BASE/'source_preparation_v2.json');prior=read(BASE/'source_preparation.json')
assert sha(BASE/'source_preparation.json')==p['original_preparation_sha256']
assert len(p['source_sha256'])==17 and len(p['unchanged_original_files'])==11
for name,h in p['source_sha256'].items():assert sha(BASE/'source_draft_v2'/name)==h
for name,h in p['original_source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==h
for name in p['unchanged_original_files']:assert p['source_sha256'][name]==p['original_source_sha256'][name]
for name in prior['unchanged_original_files']:assert p['source_sha256'][name]==prior['original_source_sha256'][name]
assert len(prior['unchanged_original_files'])==8
for name,methods in prior['unchanged_function_AST'].items():
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))) if isinstance(n,ast.FunctionDef)}
    old=functions(NEW/'independent_plant_clock_saved_root_review_v1/source_audit_v3'/name)
    new=functions(BASE/'source_draft_v2'/name)
    for method in methods:assert old[method]==new[method],(name,method)
xml=ET.parse(BASE/'root_tests_v2.xml').getroot();suites=list(xml) if xml.tag!='testsuite' else [xml]
assert sum(int(s.attrib['tests']) for s in suites)==106
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
core=NEW/'independent_plant_pending_publication_v1/source_draft_v1/clock_core.py'
assert sha(core)=='c67d970962a1fe67258154d13beb82c04a69f70c25426e77736def8c790770d1'
report=dict(passed=True,source_review_pass=True,reviewer='root',source_sha256=p['source_sha256'],source_directory=(BASE/'source_draft_v2').as_posix(),source_preparation_sha256=sha(BASE/'source_preparation_v2.json'),pending_core_sha256=sha(core),root_synthetic_tests=106,root_tests_sha256=sha(BASE/'root_tests_v2.xml'),byte_unchanged_vs_original_auditor_files=prior['unchanged_original_files'],required_clock_request_contract='pending_BUSY_same_job_max10_before_original_activation',reviewed_semantics=['All publication attempts bind same immutable job bytes and original activation deadline, with initial plus at most nine BUSY retries.','Retry guard is distinct from actual transport start/end; late publication cannot become timely admission.','Expiry must occupy next eligible tick and all events are bounded by captured/reserved terminal tick; pending tail covers completed ticks.','Unknown observer return remains one actual terminal pre-native action, with no later transport/admission/native credit.','Bound outer interruption supplements core failure; returned publication and committed event success remain separately checked.','Original strict physical, PD, full291 history, timing, command, watchdog, stage and process checks retained.'],limitations=['Saved-only verifier; no actual clock request or run selected by this review.','Original audit cannot validate retry ledgers. New actual request/launcher/helper binding still required.'],actual_clock_selected=False,actual_audit_selected=False,task_array_loads=0,model_calls=0,native_steps=0,writer_sha256=sha(__file__))
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'source_review_pass':True,'sha256':sha(OUT/'review.json'),'tests':106}))
