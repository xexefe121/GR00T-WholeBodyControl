"""Preserve the failed pre-inference run; make schedule identity portable."""
from pathlib import Path
import hashlib,json,shutil,struct
BASE=Path(__file__).resolve().parent
old=BASE/'direct_target_width251_evaluation_v1';new=BASE/'direct_target_width251_evaluation_v2'
new.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
for p in old.iterdir():
    if p.is_file() and p.suffix in ('.py','.ps1'):shutil.copy2(p,new/p.name)
shutil.copytree(old/'source_draft_v1',new/'source_draft_v1',ignore=shutil.ignore_patterns('__pycache__'))
fit=BASE/'direct_target_width251_student_v1'
request=read(fit/'training_request.json');selected=read(fit/'selected_protocol.json')
assert request['learning_rate_values']==selected['learning_rate_values']
digest=hashlib.sha256(struct.pack('<10000d',*selected['learning_rate_values'])).hexdigest()
p=new/'source_draft_v1/recovery_release.py';text=p.read_text()
before="    assert actual == selected_rates(), 'Different selected rate schedule'"
after="    # Exact selected IEEE754 values; do not recompute libm cosine on another OS.\n    import hashlib, struct\n    assert hashlib.sha256(struct.pack('<10000d', *actual)).hexdigest() == '"+digest+"', 'Different selected rate schedule'"
assert text.count(before)==1;p.write_text(text.replace(before,after),encoding='utf-8')
sources={p.relative_to(new/'source_draft_v1').as_posix():sha(p) for p in (new/'source_draft_v1').rglob('*.py')}
prep=read(old/'source_preparation.json');prep.update(source_sha256=sources,parent_preparation_sha256=sha(old/'source_preparation.json'),portability_fix='Exact selected little-endian float64 schedule SHA replaces cross-platform cosine recomputation.')
write(new/'source_preparation.json',prep)
review=read(old/'source_root_review.json');review.update(source_sha256=sources,parent_review_sha256=sha(old/'source_root_review.json'),
    selected_schedule_sha256=digest,controller_and_physics_byte_exact=True,
    failed_prior_completion_sha256=sha(old/'witness_completion_verification.json'))
for name in ('launch_helper_preparation_v2.json','powershell_path_normalization_v1.json','launch_helper_tests_v3.xml','launch_helper_tests_v4.xml'):
    if (old/name).exists():shutil.copy2(old/name,new/name)
write(new/'source_root_review.json',review)
for name in ('evaluate_direct_target_student.py','direct_runtime.py','causal_features.py','runtime_common.py','head_activation_witness.py'):
    assert sha(new/'source_draft_v1'/name)==sha(old/'source_draft_v1'/name)
print(json.dumps({'selected_schedule_sha256':digest,'changed_sources':[n for n,d in sources.items() if sha(old/'source_draft_v1'/n)!=d]}))
