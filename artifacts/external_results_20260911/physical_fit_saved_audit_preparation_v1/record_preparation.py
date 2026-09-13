"""Only compile source and inspect saved test reports; never run the real audit."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (BASE/'preparation_report.json').exists()
sources=[ROOT/'audit_physical_fit_evidence.py',ROOT/'audit_physical_fit_math.py']
for p in sources:compile(p.read_text(encoding='utf-8'),str(p),'exec')
prior_report=BASE.parent/'velocity_fit_evidence_independent_v1/report.json'
old=ROOT/'audit_velocity_fit_evidence.py'
assert sha(old)==json.loads(prior_report.read_text())['source_sha256']
suites=list(ET.parse(BASE/'tests_final.xml').getroot().iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites)==18
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
report=dict(kind='independent_saved_physical_fit_audit_preparation',passed=True,tests=18,failures=0,errors=0,skips=0,
    source_sha256={p.as_posix():sha(p) for p in sources},
    evidence_sha256={name:sha(BASE/name) for name in ('README.md','test_saved_audit.py','tests.xml','tests_final.xml','record_preparation.py')},
    prior_audit_source_sha256=sha(old),prior_completed_audit_report_sha256=sha(prior_report),
    prior_velocity_audit_unchanged=True,actual_model_evaluations=0,actual_ONNX_calls=0,optimizer_updates=0,physics_steps=0,
    real_checkpoint_or_graph_audit_not_executed=True,
    scopes=['actual training receipt SHA required','all3054 requested physical identities and statuses',
      'same-clock successor pairing and99/819/100 requested weights','all45000 private sampler calls/2880000 row-axis pairs',
      'all5000 learning rates and three-term/cell loss arithmetic at original precision',
      'ordinary75000 optimizer counters/norm/span/ONNX weights and graph',
      'all initial legacy prediction bytes to70000','all saved nominal/velocity/physical metrics including groups and fixed first24',
      'initial/final and training counters','input/source initial and final hashes','first mismatch/context/failure preservation'],
    final_training_receipt_and_fit_outputs_still_required=True)
(BASE/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,tests=18,report_sha256=sha(BASE/'preparation_report.json'),source_sha256=report['source_sha256'])))
