"""Saved source/test receipts only; no network, model or optimizer execution."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;SRC=BASE/'source_draft_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (BASE/'preparation_report_v2.json').exists()
prior=json.loads((BASE/'preparation_report.json').read_text())
for name,digest in prior['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest
for p in SRC.rglob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
suites=list(ET.parse(BASE/'preparation_tests_v2_final.xml').getroot().iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites)==32
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
derivation=json.loads((BASE/'source_derivation.json').read_text())
assert all(sha(SRC/name)==digest for name,digest in derivation['original_sources_sha256'].items())
absent=['training_request.json','training_frozen_inputs.json','training_clearance.json','fit']
assert all(not (BASE/name).exists() for name in absent)
report=dict(preparation_only=True,passed=True,tests=32,failures=0,errors=0,skips=0,
    actual_model_evaluations=0,actual_onnx_calls=0,optimizer_updates=0,physics_steps=0,
    prior_source_and_evidence_preserved=True,unchanged_original_helpers=32,
    source_sha256={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*.py')},
    evidence_sha256={name:sha(BASE/name) for name in ('preparation_report.json','PREPARATION_V2.md',
      'preservation_v2_derivation.json','prepare_preservation_v2.py','test_preparation_v2.py',
      'preparation_tests_v2_final.xml','record_preparation_v2.py')},
    max_training_rows=36315000,max_actual_head_calls=1150,
    no_training_request_or_clearance=True,absent_artifacts=absent)
(BASE/'preparation_report_v2.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,tests=32,report_sha256=sha(BASE/'preparation_report_v2.json'))))
