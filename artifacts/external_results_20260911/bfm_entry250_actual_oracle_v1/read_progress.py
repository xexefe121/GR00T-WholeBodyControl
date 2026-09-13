"""Read existing run artifacts only; never infer, optimize, or step physics."""
from pathlib import Path
import json
import numpy as np

base=Path(__file__).resolve().parent
result={'process':json.loads((base/'process_status.json').read_text(encoding='utf-8-sig'))}
progress=[]
stdout=(base/'stdout.log').read_bytes()
stdout=stdout.decode('utf-16' if stdout.startswith(b'\xff\xfe') else 'utf-8-sig')
for line in stdout.splitlines():
    try:value=json.loads(line)
    except json.JSONDecodeError:continue
    if 'control' in value:progress.append(value)
result['latest_reported_progress']=progress[-1] if progress else None
partial=base/'nominal/trace.partial.npz'
if partial.exists():
    with np.load(partial,allow_pickle=False) as a:
        result['atomic_checkpoint']=json.loads(str(a['checkpoint_metadata']))
        result['atomic_checkpoint'].update(physics_steps=len(a['physics_torque']),clock=float(a['physics_time'][-1]))
final=base/'nominal/report.json'
if final.exists():
    r=json.loads(final.read_text())
    result['final']={k:r.get(k) for k in ('requested_controls','completed_full_controls','failure','physics_steps','full_segment_completed','trace_sha256')}
    hold=base/'post_lifecycle_hold_5s/report.json'
    if hold.exists():
        r=json.loads(hold.read_text());result['hold']={k:r.get(k) for k in ('requested_controls','completed_full_controls','failure','full_segment_completed','trace_sha256')}
print(json.dumps(result))
