from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'direct_target_response_balanced_fit_independent_v3';OLD=NEW/'direct_target_response_balanced_fit_independent_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(BASE/'source_preparation.json')=='017bdb25ab313ef4359098f3a1d372364922b8bd6713896e0d712b7b79c05816'
p=read(BASE/'source_preparation.json');prior=read(NEW/'direct_target_response_fit_audit_root_review_v2/review.json')
assert len(p['source_sha256'])==13 and len(p['unchanged_source_sha256'])==11
for name,h in p['source_sha256'].items():assert sha(BASE/'source_prepared_v1'/name)==h
for name,h in p['unchanged_source_sha256'].items():assert sha(OLD/'source_prepared_v1'/name)==h
for name,h in p['helper_sha256'].items():assert sha(BASE/name)==sha(OLD/name)==prior['helper_sha256'][name]==h
for path,h in p['reference_sha256'].items():assert sha(path)==h
for name,h in p['evidence_sha256'].items():assert sha(BASE/name)==h
old=(OLD/'source_prepared_v1/audit_saved_warm.py').read_text(encoding='utf-8-sig')
new=(BASE/'source_prepared_v1/audit_saved_warm.py').read_text(encoding='utf-8-sig')
removed="            close('nominal_f32_mean',progress[:,0],wanted['nominal'],nominal=True)\n"
assert old.count(removed)==1 and old.replace(removed,'')==new
assert "close('nominal_cells',progress[:,0],cell_values['nominal'].mean(axis=1),nominal=True)" in new
assert 'rtol=3e-7 if nominal else 5e-12' in new
xml=ET.parse(BASE/'root_tests.xml').getroot();suites=list(xml) if xml.tag!='testsuite' else [xml]
assert sum(int(s.attrib['tests']) for s in suites)==50
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
report=dict(source_review_pass=True,helper_review_pass=True,reviewer='root',source_preparation_sha256=sha(BASE/'source_preparation.json'),source_sha256=p['source_sha256'],helper_sha256=p['helper_sha256'],prior_source_review_sha256=sha(NEW/'direct_target_response_fit_audit_root_review_v2/review.json'),root_synthetic_tests=50,root_tests_sha256=sha(BASE/'root_tests.xml'),single_removed_line=p['removed_single_line'],all_other_source_exact=True,reviewed_semantics=['CUDA float32 mean need not match NumPy float32 reduction order. Exact rational15-positive-cell mean confirms original GPU-versus-f64 gate passes all3000 rows.','Removed only redundant CPU-f32-order check. Original nominal3e-7 relative bound, all54response cells, physical cells/totals and1e-5rad model parity unchanged.','Literal failing row and independent positive reduction order counterexample covered; analytical gamma16 bound is explanatory and does not replace original tighter gate.','Both failed saved audits remain preserved; no model parameters, training, saved predictions or physics changed.'],actual_audit_selected=False,task_array_loads=0,task_checkpoint_loads=0,model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,writer_sha256=sha(__file__))
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'source_review_pass':True,'sha256':sha(OUT/'review.json')}))
