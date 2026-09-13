from pathlib import Path
import hashlib,json,sys,unittest,io
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'independent_plant_pending_publication_saved_actual_v1';OLD=NEW/'independent_plant_clock_saved_actual_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(BASE/'helper_preparation.json')=='20407fd5dc10b837e8c8e87b94a69e337b5e066ed52398ac392147693097af18'
p=read(BASE/'helper_preparation.json')
for name,h in p['helper_sha256'].items():assert sha(BASE/name)==h
for name,h in p['evidence_sha256'].items():assert sha(BASE/name)==h
assert sha(BASE/'verify_completion.py')==sha(OLD/'verify_completion.py')==p['original_owner_sha256']
assert sha(BASE/'preserved_saved_audit_template.ps1.txt')==sha(OLD/'preserved_saved_audit_template.ps1.txt')==p['original_template_sha256']
old=(OLD/'prepare_launch.py').read_text(encoding='utf-8-sig')
for before,after in [('Frozen source-audit-v3 launcher','Frozen retry-aware source-draft-v2 launcher'),('independent_plant_clock_saved_root_review_v1/source_audit_v3','independent_plant_pending_publication_saved_audit_v1/source_draft_v2'),('independent_plant_clock_saved_root_review_v1/root_source_review_v3.json','independent_pending_publication_saved_root_review_v1/review.json'),('278c509418ef1c7a1be655693a840217dd8c1894a6a6c33f0757a712ad8b9ac7','81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0')]:
    assert old.count(before)==1;old=old.replace(before,after)
assert old==(BASE/'prepare_launch.py').read_text(encoding='utf-8-sig')
source=NEW/'independent_plant_pending_publication_saved_audit_v1/source_draft_v2'
for name,h in p['audit_source_sha256'].items():assert sha(source/name)==h
assert sha(p['audit_source_review_path'])==p['audit_source_review_sha256']
assert read(p['audit_source_review_path'])['source_sha256']==p['audit_source_sha256']
sys.path.insert(0,str(BASE));import test_launch,prepare_launch
log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_launch))
(OUT/'root_tests.log').write_text(log.getvalue(),encoding='utf-8')
assert result.wasSuccessful() and result.testsRun==10 and not result.skipped
assert prepare_launch.render('a'*64)==(BASE/'template_preview.ps1.txt').read_text(encoding='utf-8-sig')
assert read(BASE/'template_parse.json')['passed'] is True
assert not (BASE/'request.json').exists() and not (BASE/'launch_receipt.json').exists()
report=dict(passed=True,helper_review_pass=True,reviewer='root',helper_preparation_sha256=sha(BASE/'helper_preparation.json'),helper_sha256=p['helper_sha256'],audit_source_sha256=p['audit_source_sha256'],audit_source_review_sha256=p['audit_source_review_sha256'],root_helper_tests=10,byte_identical_owner_and_template=True,launcher_only_source_namespace_changes=True,reviewed_semantics=['Exact retry-aware17-file source/review replaces original saved-clock auditor.','Original hidden durable wrapper, exact request and concrete-review clearance, known raw exit and owner completion retained.','Saved evidence integrity remains distinct from physical, timing and command qualification.','Actual completed-clock request and receipt must be prepared and checked after future clock run; no selection here.'],actual_audit_selected=False,actual_request_created=False,model_calls=0,native_steps=0,writer_sha256=sha(__file__))
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'sha256':sha(OUT/'review.json'),'tests':10}))
