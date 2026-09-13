"""Frozen preparation review only; no controller/model/native execution."""
import ast,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');OWNER=BASE/'direct_target_student_evaluation_v1';SOURCE=OWNER/'source_draft_v1';OUT=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={}
def pin(p,d=None):
    p=Path(p);actual=sha(p)
    if d is not None:assert actual==d,p
    pins[p.as_posix()]=actual
prep=OWNER/'source_preparation.json';pin(prep,'7391598abf062789b8daae00d85076470b947c0b3036242eccd8b2f5b36bab5f')
r=read(prep);assert r['passed'] and r['source_preparation_only']
assert len(r['source_sha256'])==30
for n,d in r['source_sha256'].items():pin(SOURCE/n,d);ast.parse((SOURCE/n).read_text())
for p,d in r['evidence_sha256'].items():pin(p,d)
old=read(OWNER/'original_sources.json');assert len(old)==22
for n,v in old.items():pin(v['original'],v['sha256']);assert sha(SOURCE/n)==v['sha256']
der=read(OWNER/'evaluator_derivation_v2.json');text=(SOURCE/'original_evaluator.py').read_text()
z=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='zero_parity')
text=text.replace(ast.get_source_segment(text,z),'')
for before,after in der['edits']:
    assert text.count(before)==1,before;text=text.replace(before,after)
assert text==(SOURCE/'evaluate_direct_target_student.py').read_text()
original=ast.parse((SOURCE/'original_evaluator.py').read_text());derived=ast.parse(text)
for name in ('get_state','assess'):
    a=next(n for n in original.body if getattr(n,'name',None)==name);b=next(n for n in derived.body if getattr(n,'name',None)==name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
xml=ET.parse(OWNER/'stub_tests_final_v3.xml');suites=xml.findall('.//testsuite')
assert sum(int(s.attrib['tests']) for s in suites)==30
for s in suites:assert all(int(s.attrib.get(k,0))==0 for k in ('failures','errors','skipped'))
features=read(OWNER/'saved_feature_checks_v1/report.json');assert features['passed'] and features['rows']==9904
assert [c['rows'] for c in features['cases']]==[3057,5980,867]
assert all(c['all_kept_feature_bytes_exact'] for c in features['cases'])
assert features['first_query250']['byte_exact'] and features['first_query250']['shape']==[1000]
for p,d in features['input_sha256'].items():pin(p,d)
for name in ('witness_binding.json','evaluation_binding.json','witness_process','evaluation_process','head_witness','nominal','post_lifecycle_hold_5s','pilot_outcome.json'):assert not (OWNER/name).exists(),name
pin(Path(__file__))
review=dict(kind='direct_target_adapter_source_preparation_review',source_review_pass=True,source_review_passed=True,preparation_review_pass=True,
    preparation_only=True,source_sha256=r['source_sha256'],input_sha256=pins,preparation_subject=dict(path=prep.as_posix(),sha256=sha(prep)),
    evidence=dict(original_helpers_byte_exact=22,stub_tests=30,saved_feature_rows=9904,first_query250_features_exact=True,
        full_evaluator_declared_transformation_exact=True,original_assess_get_state_AST_exact=True,
        learned_feature_contract='Pure measured state and prepared goals: original0:52 +75:1023; no BFM/history inputs.',
        output_contract='default64 + existing span32 widened64 * normalized head32 widened64, then native64 clamp.',
        phase_contract='Startup/terminal retain original BFM helper arithmetic and raw feedback. Learned phase calls only head, commits actual applied-target inverse, and maintains measured BFM history for terminal handoff.',
        commit_contract='History/prior/control counter stage until proposal checks and first-query/witness comparison pass; commit immediately before actual command.',
        failure_contract='Current input/output and staged versus committed history/counters retained; native partial exception clock and full291 preserved.',
        gate_contract='Actual final5000 checkpoint/head/report/reviews and one future WSL batch1 witness required. Exact canonical250 prefix and first query inputs/output before learned activation.',
        dtype_note='Complete mixed-phase delta array is float64; original startup float32 comparison occurs before learned rows. This does not change startup values.'),
    future_requested_main_controls=1569,future_conditional_hold_controls=250,future_separate_head_witness_calls=1,
    execution_authorized_by_this_review=False,behavioral_qualification=False,hardware_authorized=False,
    reviewer_model_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0)
for p,d in pins.items():assert sha(p)==d,p
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps(dict(source_review_pass=True,pins=len(pins),review_sha256=sha(OUT/'review.json'))))
