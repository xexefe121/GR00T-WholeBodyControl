from pathlib import Path
import hashlib,json
import numpy as np
B=Path(__file__).resolve().parent
p=B.parent/'direct_target_student_evaluation_v1/nominal/trace.npz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=sha(p)
with np.load(p,allow_pickle=False) as a:
    mode=a['controller_mode'].copy();timing=a['inference_ms'].copy()
assert len(mode)==316 and np.isfinite(timing).all()
rows={}
for key,name in [(0,'startup_BFM'),(1,'direct_target'),(2,'terminal_BFM')]:
    x=timing[mode==key]
    rows[name]=dict(controls=len(x),p50_p95_max_ms=np.percentile(x,[50,95,100]).tolist() if len(x) else None,
                    policy_only_20ms_misses=int(np.sum(x>20)))
assert sha(p)==digest
report=dict(trace_sha256=digest,source_sha256=sha(Path(__file__)),phases=rows,new_inference_calls=0,native_steps=0,
            limitation='Saved proposal/inference timer only; excludes independent wall-clock plant deadline and complete input/IPC/control loop. No real-time qualification.')
out=B/'phase_timing.json';assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
