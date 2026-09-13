from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def stats(a):return dict(zip(('p50','p95','p99','max'),map(float,np.percentile(a,[50,95,99,100]))))
fields=json.loads((OUT/'field_inspection.json').read_text(encoding='utf-8-sig'))
trace=NEW/'independent_plant_clock_timeout_correction_v1/run/trace.npz'
assert sha(trace)==fields['trace_sha256']
with np.load(trace,allow_pickle=False) as a:
    ns=a['step_nominal_start'];ne=a['step_nominal_end'];starts=a['step_actual_start'];ends=a['step_actual_end']
    assert all(v.shape==(4694,) for v in (ns,ne,starts,ends))
    assert np.all(ne-ns==2000000) and np.all(np.diff(ns)==2000000)
    assert np.all(ends>=starts) and np.all(starts>=ns)
    wake=starts-ns;body=ends-starts;late=np.maximum(ends-ne,0)
    indices=np.flatnonzero(late);assert indices.tolist()==[610,3235,4250,4669,4670,4671]
    rows=[dict(index=int(i),control_boundary=bool(i%10==0),wake_delay_ns=int(wake[i]),body_ns=int(body[i]),deadline_overrun_ns=int(late[i]),wake_already_overdue=bool(starts[i]>ne[i])) for i in indices]
    nearby=[dict(index=int(i),wake_delay_ns=int(wake[i]),body_ns=int(body[i]),deadline_overrun_ns=int(late[i])) for i in range(608,613)]
    result=dict(saved_timing_analysis_complete=True,trace_sha256=sha(trace),saved_audit_sha256=fields['audit_sha256'],returned_steps=4694,deadline_misses=6,misses=rows,first_miss_neighbors=nearby,all_steps=dict(wake_delay_ns=stats(wake),body_ns=stats(body)),boundary_steps=dict(wake_delay_ns=stats(wake[np.arange(len(wake))%10==0]),body_ns=stats(body[np.arange(len(body))%10==0])),nonboundary_steps=dict(wake_delay_ns=stats(wake[np.arange(len(wake))%10!=0]),body_ns=stats(body[np.arange(len(body))%10!=0])),limitations=['Saved timestamp attribution only; body combines Python boundary/history/mailbox, native step/capture/check and OS preemption.','Wake delay alone does not identify hypervisor/OS scheduling cause. No replay or fresh clock run performed.'],model_calls=0,native_steps=0,writer_sha256=sha(__file__))
    with (OUT/'report.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'sha256':sha(OUT/'report.json'),'misses':rows,'all_steps':result['all_steps']}))
