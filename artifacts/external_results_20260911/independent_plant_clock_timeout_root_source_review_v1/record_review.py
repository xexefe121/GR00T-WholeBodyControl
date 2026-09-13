"""Source-only review of stage deadlines and durable clock evidence."""
import hashlib,json,ast,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'independent_plant_clock_timeout_correction_v1'
OLD=HERE.parent/'independent_plant_process_clock_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep=BASE/'source_preparation_v1.json'
assert sha(prep)=='9e9fcba3f0ea2c6aee64b9163a869933acaee72bd780bbfdc3b1ade796e5de36'
p=read(prep);assert p['source_preparation_passed'] is True
for relative,digest in p['source_sha256'].items():assert sha(BASE/relative)==digest,relative
for name in p['unchanged_runner_files']:
    assert (BASE/'source_draft_v1'/name).read_bytes()==(OLD/'source_runner_v1'/name).read_bytes(),name
assert len(p['unchanged_runner_files'])==17
for path,digest in p['evidence_sha256'].items():assert sha(path)==digest,path
assert (BASE/'prepare_stage_preserved_template.py').read_bytes()==(OLD/'prepare_stage_preserved_template.py').read_bytes()
def loops(path):return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.While)]
assert loops(BASE/'source_draft_v1/run_clock.py')==loops(OLD/'source_runner_v1/run_clock.py')
tests=ET.parse(BASE/'stub_tests_v2.xml').getroot().findall('testsuite')
assert sum(int(x.get('tests','0')) for x in tests)==52
assert all(int(x.get(k,'0'))==0 for x in tests for k in ('failures','errors','skipped'))
out=HERE/'review.json'
v=dict(passed=True,source_review_pass=True,preparation_only=True,
    source_preparation_subject=dict(path=str(prep),sha256=sha(prep)),source_sha256=p['source_sha256'],
    unchanged_runner_files=p['unchanged_runner_files'],unchanged_hot_loop_AST=True,
    reviewed=['Root read complete watchdog, runner, authorization delta, durable journal/counters and new tests.',
        'Root read complete outer launcher, diagnostic verdict and completion helper; original hidden template byte-identical.',
        'Setup240s, fixed epoch plus120s plant and180s preservation are distinct; outer555s+5s fallback retained.',
        'Original2ms/20ms deadlines,60s elapsed,100step debt,18190 native and4 serializations unchanged.',
        'Epoch record precedes hot loop and a second fixed-epoch check rejects slow allocation/fsync without rebasing.',
        'Immediate exit counters precede cleanup; native return uncertainty remains visible; hard timeout cannot qualify without final evidence.'],
    synthetic_tests=p['tests'],budget_proposal=p['proposed_watchdog_budgets'],
    actual_clock_selected=False,actual_native_steps=0,actual_serializations=0,actual_model_calls=0,
    remaining='Concrete actual request/launcher and independent saved-stage evidence review remain necessary before one separately selected clock run.')
with out.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'sha256':sha(out)}))
