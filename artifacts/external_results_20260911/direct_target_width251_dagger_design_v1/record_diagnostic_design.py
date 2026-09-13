"""Bind the additional requested source/text diagnostic scope; no task data."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior=BASE/'report.json';report=json.loads(prior.read_text())
report.update(prior_design_report=dict(path=prior.as_posix(),sha256=sha(prior)),
    consistency_design=dict(path=(BASE/'CONSISTENCY_DIAGNOSTIC.md').as_posix(),sha256=sha(BASE/'CONSISTENCY_DIAGNOSTIC.md')),
    minimum_conditional_diagnostic=dict(selected=False,old_nominal_rows=9904,old_physical_rows=3054,new_rows=1018,
        exact_feature_alias_rows=13976,proximity_new_queries=72,proximity_old_candidates=12958,
        actual_arrays_loaded=0,exact_current_only_aliases_separate=True,physical_targets_absolute=True,
        optional_pre_fit_inference=dict(selected=False,backend='qualified81000 FP64 ORT CPU',batch=256,calls=4,rows=1018,
            gradients=0,optimizer_updates=0,native_steps=0),
        warning='MPC warm/committed plans can make labels multi-valued under the current1323 features; no automatic averaging or dropping.'),
    additional_source_sha256=sha(Path(__file__)))
with (BASE/'report_v2.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(report=(BASE/'report_v2.json').as_posix(),sha256=sha(BASE/'report_v2.json'))))
