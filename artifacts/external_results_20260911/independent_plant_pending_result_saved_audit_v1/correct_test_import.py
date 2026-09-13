"""Preserve unexecuted malformed copied import prefix, then make it local."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;p=B/'source_draft_v2/test_cross_job_iteration.py'
raw=p.read_bytes();saved=B/'test_cross_job_iteration_preserved_invalid_import.py.txt'
with saved.open('xb') as f:f.write(raw)
text=raw.decode();line="    str(Path(__file__).resolve().parents[1] / 'independent_plant_pending_result_saved_audit_v1/source_draft_v1')))\n"
assert text.count(line)==1
p.write_text(text.replace(line,''),newline='\n')
with (B/'test_import_correction_v2.json').open('x') as f:
    json.dump({'kind':'source_syntax_preflight_correction','before_sha256':hashlib.sha256(raw).hexdigest(),
      'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'preserved_path':str(saved),
      'cause':'Reviewer changed import selector to multiline before the copied-test derivation; one continuation line survived removal.',
      'actual_task_calls':0,'tests_executed_on_malformed_copy':0,'runtime_audit_math_changed':False},f,indent=2)
