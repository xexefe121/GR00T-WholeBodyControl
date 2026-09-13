from pathlib import Path
import ast, hashlib, json, xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent; NEW=OUT.parent
BASE=NEW/'direct_target_width512_fit_independent_v1'
PRIOR=NEW/'direct_target_response_balanced_fit_independent_v3'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def txt(p): return Path(p).read_text(encoding='utf-8-sig')
assert sha(BASE/'source_preparation.json')=='d252b0a8875636601d90bf538f342097043bce86666d6ddb23c4d708e00f0d2e'
p=read(BASE/'source_preparation.json'); src=BASE/'source_prepared_v1'; old=PRIOR/'source_prepared_v1'
assert len(p['source_sha256'])==16 and len(p['unchanged_source_sha256'])==9 and len(p['helper_sha256'])==4
for name,h in p['source_sha256'].items():
    assert sha(src/name)==h
    ast.parse(txt(src/name))
for name,h in p['unchanged_source_sha256'].items(): assert sha(old/name)==h
for name,h in p['helper_sha256'].items(): assert sha(BASE/name)==h
for path,h in p['reference_sha256'].items(): assert sha(path)==h
for name,h in p['evidence_sha256'].items(): assert sha(BASE/name)==h
for name,h in p['producer_source_sha256'].items(): assert sha(Path(p['experiment'])/'source_snapshot_v1'/name)==h
assert txt(old/'audit_graph.py').replace('[(256,1323),(256,256),(23,256)]','[(512,1323),(512,512),(23,512)]')==txt(src/'audit_graph.py')
assert sha(BASE/'verify_completion.py')==sha(PRIOR/'verify_completion.py')
for name in ('freeze_launch.py','run_audit_durable.ps1'):
    assert txt(PRIOR/name).replace('saved_response_balanced_warm_only','saved_width512_warm_only')==txt(BASE/name)
main=txt(src/'audit_saved_warm.py')
assert 'nominal_f32_mean' not in main
assert "close('nominal_cells',progress[:,0],cell_values['nominal'].mean(axis=1),nominal=True)" in main
assert 'rtol=3e-7 if nominal else 5e-12' in main
assert "qualification[condition]=maximum<=1e-5" in main
assert "check(condition+'_initial_drift_gate',maximum<=1e-5)" in main
xml=ET.parse(BASE/'root_tests.xml').getroot(); suites=list(xml) if xml.tag!='testsuite' else [xml]
assert sum(int(s.attrib['tests']) for s in suites)==77
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
report=dict(source_review_pass=True,helper_review_pass=True,reviewer='root',source_preparation_sha256=sha(BASE/'source_preparation.json'),source_sha256=p['source_sha256'],helper_sha256=p['helper_sha256'],prior_source_review_sha256=sha(NEW/'direct_target_response_fit_audit_root_review_v3/review.json'),root_synthetic_tests=77,root_tests_sha256=sha(BASE/'root_tests.xml'),unchanged_source_count=9,reviewed_semantics=[
    'Independent explicit old/new tensor concatenation reconstructs all six actor tensors, every warm moment, whole AdamW group, shared step6000, local generator state and unchanged source/global RNG without constructing a model or optimizer.',
    'Full10000 schedule independently reconstructs all8.64M pair slots and exact original3000 prefix; original context chronology and normalization arithmetic remain byte-identical.',
    'Warm6000-to16000 scalar steps,250-update inclusive ramp and9750-update cosine, width512 FP64 graph shapes and exact30000/146860000 training counters match frozen producer schema.',
    'Original nominal3e-7 relative gate and other5e-12 reductions retained; invalid NumPy-f32 reduction-order check remains removed. Original initial1e-5 and nine finalFP64 gates unchanged; finalGPU32 drift diagnostic only.',
    'Sixteen direct release subjects, owner/pre-post pin and process accounting retained. Saved algebra cannot prove output or gradient authenticity, or closed-loop stability.'
],actual_audit_selected=False,task_array_loads=0,task_checkpoint_loads=0,model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,synthetic_model_forwards=0,writer_sha256=sha(__file__))
with (OUT/'review.json').open('x',encoding='utf-8') as f: json.dump(report,f,indent=2); f.write('\n')
print(json.dumps(dict(source_review_pass=True,sha256=sha(OUT/'review.json'))))
