import hashlib,json,sys,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'direct_target_response_balanced_fit_independent_v1'
OLD=NEW/'direct_target_context_pair_fit_independent_v1/source_draft_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(BASE/'source_preparation.json')=='42e543d4b73346164c9826abf32cf279babc3c6246ecc019f6cdb43c9dcfa21b'
p=read(BASE/'source_preparation.json');source=Path(p['source_directory'])
assert p['source_preparation_passed'] is True and len(p['source_sha256'])==11 and p['synthetic_tests_passed']==39
for name,digest in p['source_sha256'].items():assert sha(source/name)==digest
for name,digest in p['copied_math_sha256'].items():assert sha(OLD/name)==sha(source/name)==digest
assert len(p['copied_math_sha256'])==5
for name,digest in p['helper_sha256'].items():assert sha(BASE/name)==digest
for path,digest in p['schema_reference_sha256'].items():assert sha(path)==digest,path
for name,digest in p['evidence_sha256'].items():assert sha(BASE/name)==digest,name
run=subprocess.run([sys.executable,'-m','pytest','--rootdir=.','--confcutdir=.','-q','--junitxml='+str(OUT/'root_tests.xml')],cwd=source,capture_output=True,text=True)
(OUT/'root_tests.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
assert run.returncode==0,run.stdout+run.stderr
xml=ET.parse(OUT/'root_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
assert sum(int(s.attrib['tests']) for s in suites)==39
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
report=dict(source_review_pass=True,helper_review_pass=True,reviewer='root',
 source_preparation_sha256=sha(BASE/'source_preparation.json'),source_sha256=p['source_sha256'],helper_sha256=p['helper_sha256'],
 copied_qualified_math_files=5,root_synthetic_tests=39,root_tests_sha256=sha(OUT/'root_tests.xml'),
 reviewed_semantics=['single corrected warm71000 fit; all16 release subjects explicitly bound',
 'exact source and initial actor/optimizer/RNG/norm comparison, six warm steps3000 and final6000',
 'unchanged original chronology, physical prior/history, normalization, data identities and fixed sampler reconstruction',
 'fixed energy weights independently reconstructed and used only in added balanced54 metrics; original metrics preserved',
 'training cell/order/totals,3000schedule/9000calls/44058000rows and five full diagnostic partitions checked',
 'same-weight FP64 graph and nine saved parity/drift comparisons; original1e-5rad gates retained',
 'actual top-level concrete review schema and safe carried-request construction covered by regressions',
 'current/prerun/postrun identities, saved owner and process linkage required; evidence versus export qualification separate',
 'request writer refuses incomplete optimization/diagnostics and selects corrected fitv2 only'],
 limitations=['Saved outputs and cell algebra checked without new model forwards or recomputed optimizer gradients.',
 'Source review does not launch the audit or qualify simulation; exact completed request/launcher still required.'],
 actual_audit_selected=False,task_arrays_loaded=0,task_checkpoint_loads=0,model_calls=0,ORT_calls=0,
 gradient_calls=0,optimizer_updates=0,native_steps=0,writer_sha256=sha(__file__))
path=OUT/'review.json'
with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(source_review_pass=True,review_path=path.as_posix(),sha256=sha(path),tests=39)))
