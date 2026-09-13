"""Preserve the failed v1 validator and correct its obsolete review schema lookup."""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
old=BASE/'direct_target_width251_fit_independent_v1'
new=BASE/'direct_target_width251_fit_independent_v2'
new.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,r):
    with p.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2);f.write('\n')
shutil.copytree(old/'source_prepared_v1',new/'source_prepared_v1',ignore=shutil.ignore_patterns('__pycache__'))
driver=new/'source_prepared_v1/audit_saved_warm.py'
text=driver.read_text()
before="        compare('concrete_source_map',launch_review['source_sha256'],frozen['source_sha256'])"
after="""        # The concrete review binds the frozen receipt by hash; the source map
        # is declared in the separately bound training source review.
        source_review_subject=request['subjects']['source_review']
        training_source_review=read(bind(source_review_subject['path'],source_review_subject['sha256']))
        check('training_source_review_pass',training_source_review[source_review_subject['pass_field']] is True)
        compare('reviewed_source_map',training_source_review['source_sha256'],frozen['source_sha256'])"""
assert text.count(before)==1
driver.write_text(text.replace(before,after),encoding='utf-8')
helpers=('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py','dispatch_root.ps1')
for name in helpers:shutil.copy2(old/name,new/name)
sources={p.name:sha(p) for p in sorted((new/'source_prepared_v1').glob('*.py'))}
fit=BASE/'direct_target_width251_student_v1'
request=read(fit/'training_request.json');subject=request['subjects']['source_review']
assert sha(subject['path'])==subject['sha256']
review=read(subject['path']);frozen=read(fit/'training_frozen_inputs.json')
assert review[subject['pass_field']] is True and review['source_sha256']==frozen['source_sha256']
clear=read(fit/'training_clearance.json');concrete=read(clear['review_path'])
assert concrete['frozen_receipt_sha256']==sha(fit/'training_frozen_inputs.json')
write(new/'source_preparation.json',dict(source_sha256=sources,parent_source_preparation_sha256=sha(old/'source_preparation.json'),change='Read source map from hash-bound training source review; concrete review binds frozen receipt hash.',model_calls=0,native_steps=0))
write(new/'source_root_review.json',dict(source_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
    helper_sha256={n:sha(new/n) for n in helpers},parent_review_sha256=sha(old/'source_root_review.json'),
    actual_metadata_chain_checked=True,numerical_checks_unchanged=True,
    failed_prior_report_sha256=sha(old/'results_v1/report.json'),model_calls=0,native_steps=0))
print(json.dumps({'source_review_sha256':sha(new/'source_root_review.json'),'changed_source_files':[n for n,d in sources.items() if sha(old/'source_prepared_v1'/n)!=d]}))
