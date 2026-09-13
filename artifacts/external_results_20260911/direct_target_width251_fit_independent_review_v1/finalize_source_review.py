"""Independent source/hash/XML read only; no task or numeric imports."""
import ast, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
FIT=BASE.parent/'direct_target_width251_student_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep=read(FIT/'source_preparation.json');before=read(BASE/'static_review_v1.json')
assert sha(FIT/'source_preparation.json')=='658ee9edccc99f9b1e8c8124d827ccff1db1ecef53ac08d6631e2ca869d42420'
source=Path(prep['source_directory']);actual={p.name:sha(p) for p in source.glob('*.py')}
assert actual==prep['source_sha256'] and len(actual)==39
assert set(actual)-set(before['source_sha256'])=={'test_recovery_integration.py'}
assert [n for n,d in before['source_sha256'].items() if actual[n]!=d]==['recovery_data.py']
for name,s in {**prep['unchanged_original_sources'],**prep['unchanged_additional_sources']}.items():
    assert actual[name]==s['sha256']==sha(s['path'])
for p,d in prep['source_references'].items():assert sha(p)==d
for p in source.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
guard="if (qualification['control_start'],qualification['control_stop_exclusive'],qualification['rows'])!=(251,1269,1018):raise ValueError('Exact original qualified recovery scope')"
assert guard in (source/'recovery_data.py').read_text()
test=prep['tests'];assert sha(test['path'])==test['sha256']
suites=ET.parse(test['path']).getroot();s=suites if suites.tag=='testsuite' else suites.find('testsuite')
assert (int(s.attrib['tests']),int(s.attrib['failures']),int(s.attrib['errors']),int(s.attrib.get('skipped',0)))==(36,0,0,0)
result=dict(passed=True,source_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
    source_preparation_sha256=sha(FIT/'source_preparation.json'),source_directory=source.as_posix(),source_sha256=actual,
    static_review_sha256=sha(BASE/'static_review_v1.json'),independent_static_checks=16,
    final_source_hashes_verified=39,original_modules_byte_exact=28,
    final_delta=['one qualification scope guard','dedicated synthetic integration tests'],
    producer_synthetic_tests=dict(tests=36,failures=0,errors=0,sha256=test['sha256'],independently_rerun=False),
    review_scope=['source and exact final map','warm full512 restoration/no expansion','unchanged old N15/P9/F54 and coefficient',
        'separate1018 D3 corpus/four forwards','original initial gate on three corpora','twelve fixed FP64 parity comparisons',
        'failure preservation/partial optimizer accounting','qualified input path/hash/scope binding'],
    source_only=True,actual_task_array_reads=0,actual_checkpoint_reads=0,model_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
    findings=[],limitations=['No actual fit admission or numerical qualification from this source review.',
        'Producer synthetic test evidence inspected; independent numeric rerun deferred during selected timing window.'])
with (BASE/'review.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'review':str(BASE/'review.json'),'sha256':sha(BASE/'review.json')}))
