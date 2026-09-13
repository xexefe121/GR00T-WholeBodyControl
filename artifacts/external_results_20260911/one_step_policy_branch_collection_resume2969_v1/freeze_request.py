"""Freeze continuation sources and every original failed-prefix evidence file."""
import ast,json,hashlib,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'one_step_policy_branch_collection_v1';SRC=BASE/'source_snapshot_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')
assert not SRC.exists() and not (BASE/'request.json').exists()
tests=read(BASE/'resume_tests.json');assert tests['passed'] and tests['tests']==5
for key,digest in tests['source_sha256'].items():assert sha(BASE/'draft'/key)==digest,key
for p in (BASE/'draft').rglob('*.py'):
    ast.parse(p.read_text());dest=SRC/p.relative_to(BASE/'draft');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
request=read(OLD/'request.json');inputs=dict(request['input_sha256']);paths=dict(request['paths'])
paths.update(prefix_data=(OLD/'collection/data').as_posix(),prefix_failure=(OLD/'collection/failure.json').as_posix(),
    prefix_verification=(OLD/'failed_prefix_verification.json').as_posix(),prefix_native_rows=(OLD/'collection/native_rows.jsonl').as_posix())
for p in (OLD/'collection').rglob('*'):
    if p.is_file():inputs[p.as_posix()]=sha(p)
for p in (OLD/'source_snapshot_v1').rglob('*.py'):inputs[p.as_posix()]=sha(p)
extras=[OLD/'request.json',OLD/'clearance.json',OLD/'launch_receipt.json',OLD/'process_status.json',OLD/'failed_prefix_verification.json',OLD/'verify_failed_prefix.py',
    BASE.parent/'one_step_failed_prefix_independent_review_v1/review.json',BASE/'derive_resume.py',BASE/'derivation.json',BASE/'test_resume.py',
    BASE/'resume_tests.json',BASE/'resume_tests.log',BASE/'read_progress.ps1',BASE/'run_durable.ps1',Path(__file__)]
for p in extras:inputs[p.as_posix()]=sha(p)
for p,digest in inputs.items():assert sha(p)==digest,p
sources={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*.py')}
request.update(kind='selected_prefix_preserving_continuation_of_same_one_control_collection',source_directory=SRC.as_posix(),source_sha256=sources,input_sha256=inputs,paths=paths,
    continuation_attempt=2,prefix_nominal_rows=2969,prefix_native_steps=29690,current_attempt_native_step_ceiling=31390,
    original_request_sha256=sha(OLD/'request.json'),prefix_manifest_sha256=sha(OLD/'collection/data/manifest.json'),
    committed_prefix='All original arrays and native ledger copied and hash-verified; first2969 nominal rows never execute again. Restore aggregate nativeattempted/returned29690 before remaining85 nominal rows, then originalfixed3054 policyorder. Graph countsstart0.',
    metadata_retry='Only atomic replacement of fully written same JSON/NPZ temp file retries PermissionError, max41 replacement attempts/40x50ms waits. No dynamics/inference retry; exhausted metadata retry preserves failure. Monitor readers use FILE_SHARE_READ|WRITE|DELETE.',
    original_attempt_immutable=True)
write(BASE/'request.json',request)
write(BASE/'clearance_DRAFT.json',dict(approved=False,request_sha256=sha(BASE/'request.json'),launcher_sha256=sha(BASE/'run_durable.ps1'),
    rows=3054,native_step_ceiling=61080,total_graph_call_ceiling=12216,prefix_native_steps=29690,current_attempt_native_step_ceiling=31390,review_path=None,review_sha256=None))
print(json.dumps(dict(request_sha256=sha(BASE/'request.json'),sources=len(sources),inputs=len(inputs),launcher_sha256=sha(BASE/'run_durable.ps1'))))
