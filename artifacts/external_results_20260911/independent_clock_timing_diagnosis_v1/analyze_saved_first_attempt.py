"""Read completed clock timestamps only; no clock, model or native execution."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stats(a):return {k:float(v) for k,v in zip(('p50','p95','p99','max'),np.percentile(a,[50,95,99,100]))}
audit_path=NEW/'independent_plant_clock_saved_actual_v1/results_v1/report.json'
assert sha(audit_path)=='48e1a6275a70998da3a2dfdda6f017b5a488c8ff4bd422d3329b4f98267ec8b1'
audit=read(audit_path);assert audit['evidence_integrity_passed'] is True
trace=NEW/'independent_plant_clock_timeout_correction_v1/run/trace.npz'
assert sha(trace)==audit['input_sha256'][trace.as_posix()]
with np.load(trace,allow_pickle=False) as a:
    starts=a['step_actual_start'];ends=a['step_actual_end']
    names=[n for n in a.files if n.startswith('step_') and ('nominal' in n or 'deadline' in n)]
    print(json.dumps({'timestamp_fields':names,'steps':len(starts)}))
    # Recorded nominal intervals are inspected below before any conclusion.
    result={'trace_sha256':sha(trace),'audit_sha256':sha(audit_path),'steps':len(starts),'body_ns':stats(ends-starts),'available_nominal_fields':names,'writer_sha256':sha(__file__),'native_steps':0,'model_calls':0}
    for name in names:
        value=a[name]
        if value.shape==starts.shape:
            result[name]={'first':int(value[0]),'last':int(value[-1])}
    with (OUT/'field_inspection.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
