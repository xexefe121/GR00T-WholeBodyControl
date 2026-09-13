"""Plot immutable measured and saved expert records; no inference/physics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parent
N=B.parent
paths=[N/'direct_target_student_evaluation_v1/nominal/trace.npz',
       N/'velocity_chord_student_v1/generation/centers.npz',
       N/'direct_target_independent_physics_v1/report.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={str(p):sha(p) for p in paths}
with np.load(paths[0],allow_pickle=False) as a:
    a={k:a[k].copy() for k in ['physics_time','physics_qpos','physics_qvel','qpos','qvel','target','global_control','physics_substeps']}
with np.load(paths[1],allow_pickle=False) as c:
    c={k:c[k].copy() for k in ['qpos','qvel','expert_target','control','dataset']}
report=json.loads(paths[2].read_text())
assert report['recorded_trace_reproduced_through_last_sample']
ids=np.flatnonzero((a['global_control']>=250)&(a['global_control']<1269))
controls=a['global_control'][ids]
centers=2038+controls-250
assert np.array_equal(c['control'][centers],controls)
assert np.all(c['dataset'][centers]==2)
t=a['physics_time']
start_steps=np.r_[0,np.cumsum(a['physics_substeps'])[:-1]]
tc=t[start_steps[ids]]
mask=t>=tc[0]
blue='#3366aa';orange='#d07820';green='#26805a'
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
ax=axes[0,0]
ax.plot(t[mask],a['physics_qpos'][mask,11],color=blue,label='Actual joint')
ax.step(tc,a['target'][ids,4],where='post',color=orange,label='Applied target')
ax.plot(tc,c['qpos'][centers,11],color=green,label='Saved expert joint')
ax.axhline(.5236,color='#aa3333',linestyle='--',label='Native upper bound')
ax.set(title='Left ankle pitch',ylabel='Position (rad)')
ax.legend(fontsize=8)
ax=axes[0,1]
ax.plot(t[mask],a['physics_qvel'][mask,10],color=blue,label='Actual velocity')
ax.plot(tc,c['qvel'][centers,10],color=green,label='Saved expert velocity')
ax.set(title='Left ankle pitch velocity',ylabel='Velocity (rad/s)')
ax.legend(fontsize=8)
ax=axes[1,0]
ax.plot(t[mask],a['physics_qpos'][mask,2],color=blue,label='Actual pelvis')
ax.plot(tc,c['qpos'][centers,2],color=green,label='Saved expert pelvis')
ax.set(title='Pelvis height',ylabel='Height (m)')
ax.legend(fontsize=8)
ax=axes[1,1]
error=a['target'][ids]-c['expert_target'][centers]
ax.plot(tc,np.sqrt(np.mean(error**2,axis=1)),color=orange)
ax.set(title='Target error against same-clock saved expert',ylabel='23-joint RMS error (rad)')
for ax in axes.flat:
    ax.set_xlabel('Original simulation time (s)')
    ax.grid(alpha=.2)
    ax.axvline(t[-1],color='#aa3333',alpha=.4)
fig.suptitle('Direct target trial: acquisition failed at 6.316 s\nSaved expert comparison uses a different trajectory after 5.000 s; no replanning',fontsize=12)
out=B/'acquisition_failure.png'
assert not out.exists()
fig.savefig(out,dpi=160)
plt.close(fig)
assert all(sha(Path(p))==h for p,h in pins.items())
receipt=dict(source_sha256=sha(Path(__file__)),input_pins=pins,image_sha256=sha(out),
             learned_controls=len(ids),native_steps=0,inference_calls=0,
             limitation='Same-clock expert is a separate trajectory, not a new expert command at the actual state.')
(B/'plot_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(out)
