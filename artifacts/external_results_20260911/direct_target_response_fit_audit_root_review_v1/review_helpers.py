import hashlib,json,sys,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
OUT=Path(__file__).resolve().parent;BASE=OUT.parent/'direct_target_response_balanced_fit_independent_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(OUT/'review.json')=='2e521adb4a89689bb7e06cf98c7d7389ef52c586f692e4aa45bba744a0bd6ce6'
assert sha(BASE/'launch_helper_preparation.json')=='5df044e5c2ad8fbe53c5484e15c5d351516be6095838aedb2f905ae2c5ddb15f'
p=read(BASE/'launch_helper_preparation.json');review=read(OUT/'review.json')
assert p['source_preparation_passed'] is True and p['synthetic_tests_passed']==8 and p['powershell_AST_passed'] is True
assert p['source_preparation_sha256']==review['source_preparation_sha256']==sha(BASE/'source_preparation.json')
for name,digest in p['helper_sha256'].items():assert sha(BASE/name)==digest
for name,digest in p['evidence_sha256'].items():assert sha(BASE/name)==digest
assert set(p['helper_sha256'])=={'prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py'}
assert p['helper_sha256']['prepare_audit_request.py']==review['helper_sha256']['prepare_audit_request.py']
run=subprocess.run([sys.executable,'-m','pytest','--rootdir=.','--confcutdir=.','-q','test_launch_helpers.py','--junitxml='+str(OUT/'root_helper_tests.xml')],cwd=BASE,capture_output=True,text=True)
(OUT/'root_helper_tests.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
assert run.returncode==0,run.stdout+run.stderr
xml=ET.parse(OUT/'root_helper_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
assert sum(int(s.attrib['tests']) for s in suites)==8
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
combined=dict(review)
combined.update(helper_review_pass=True,helper_sha256=p['helper_sha256'],helper_preparation_sha256=sha(BASE/'launch_helper_preparation.json'),
 prior_source_review_sha256=sha(OUT/'review.json'),root_helper_tests=8,writer_sha256=sha(__file__),
 reviewed_launcher_semantics=['freezer requires literal four-helper hashes from source review',
 'exact selected request, source and completed-owner subjects pinned before hidden child',
 'CreateNew one-shot lock, acquired process handle, raw known exit and complete pre/post hashes preserved',
 'saved evidence success distinct from export success; owner verifies absence, request/launch/PID and full output hashes',
 'actual request and launch still require separate concrete root review before selected dispatch'])
path=OUT/'combined_review.json'
with path.open('x',encoding='utf-8') as f:json.dump(combined,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,path=path.as_posix(),sha256=sha(path))))
