import hashlib,json,sys,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'direct_target_causal_response_evaluation_v1';OLD=NEW/'direct_target_causal_context_evaluation_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(BASE/'launch_helper_preparation_v2.json')=='d20b5537223f4740df79aa540e0e9dff85d864c82cf830dda5d9730b757e056e'
p=read(BASE/'launch_helper_preparation_v2.json');source=read(BASE/'source_preparation.json')
assert p['source_sha256']==source['source_sha256'] and len(p['helper_sha256'])==8
for name,digest in p['helper_sha256'].items():assert sha(BASE/name)==digest,name
for name,digest in source['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest,name
same=['prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py']
for name in same:assert sha(BASE/name)==sha(OLD/name),name
review_path=NEW/'direct_target_causal_response_evaluation_independent_review_v1/review.json'
assert sha(review_path)=='e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121'
independent=read(review_path);assert independent['source_sha256']==source['source_sha256']
assert independent.get('passed',independent.get('source_review_pass')) is True
run=subprocess.run([sys.executable,'-m','pytest','--rootdir=.','--confcutdir=.','-q','test_launch_helpers.py','test_release_helpers.py','--junitxml='+str(OUT/'root_tests.xml')],cwd=BASE,capture_output=True,text=True)
(OUT/'root_tests.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
assert run.returncode==0,run.stdout+run.stderr
xml=ET.parse(OUT/'root_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
assert sum(int(s.attrib['tests']) for s in suites)==41
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
review=dict(passed=True,helper_review_pass=True,source_review_pass=True,reviewer='root',helper_sha256=p['helper_sha256'],source_sha256=source['source_sha256'],
 helper_preparation_sha256=sha(BASE/'launch_helper_preparation_v2.json'),source_preparation_sha256=sha(BASE/'source_preparation.json'),
 independent_runtime_source_review_sha256=sha(review_path),root_helper_tests=41,unchanged_four_helpers=same,
 reviewed_semantics=['single causal71000 final subject bound to corrected fitv2 and16 exact release/data/energy roles',
 'source/fit-owner/independent-fit-audit/final-release/helper reviews required before model-ready binding',
 'original1569 main controls and conditional250 hold, one WSL witness call, complete full291/query250 gates preserved',
 'original launcher, diagnostic verdict and owner code byte-identical; explicit corrected helper-preparation version pinned',
 'runtime inventory excludes training corpora and retains actual native/reference/runtime identities'],
 actual_endpoint_selected=False,actual_binding_created=False,model_calls=0,native_steps=0,writer_sha256=sha(__file__))
path=OUT/'review.json'
with path.open('x',encoding='utf-8') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,path=path.as_posix(),sha256=sha(path))))
