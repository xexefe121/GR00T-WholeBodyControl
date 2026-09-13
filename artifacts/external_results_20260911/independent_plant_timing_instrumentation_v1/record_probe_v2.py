"""Freeze corrected pure probe and fake tests, preserving v1 preparation."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE/'source_preparation.json')=='9c791af95c3a74bcb3618807393e2f79cd0d8eed9c0a83d879dc7691f8c530e4'
log=(BASE/'probe_v2_tests.log').read_text();assert 'Ran 31 tests' in log and log.rstrip().endswith('OK')
report=dict(passed=True,source_preparation_only=True,source_sha256={p.name:sha(p) for p in SOURCE.glob('*.py')},
    prior_preparation_sha256=sha(BASE/'source_preparation.json'),
    evidence_sha256={name:sha(BASE/name) for name in ('PROBE_V2.md','probe_v1_to_v2.diff','probe_v2_derivation.json','derive_probe_v2.py','probe_v2_tests.log','record_probe_v2.py')},
    tests=31,failures=0,errors=0,skips=0,runtime_source_changes=0,actual_clock_calls=0,
    actual_gc_callbacks_installed=0,worker_processes=0,native_steps=0,model_calls=0,actual_run_selected=False)
with (BASE/'source_preparation_v2.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation_v2.json'),source_sha256=report['source_sha256'])))
