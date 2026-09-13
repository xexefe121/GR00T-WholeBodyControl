import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
with np.load(BASE/'intent_retarget_v2/pico/kinematic_diagnostics.npz',allow_pickle=False) as z:
    v2={key:z[key].copy() for key in z.files}
with np.load(BASE/'intent_retarget_v3/pico/kinematic_diagnostics.npz',allow_pickle=False) as z:
    v3={key:z[key].copy() for key in z.files}
indices={0,1,360,361}
crossings=[]
for channel,key in ((0,'original_task_error'),(1,'original_task_error'),(0,'foot_position_error'),(1,'foot_position_error')):
    for threshold in (.02,.05,.1,.3):
        where=np.flatnonzero(v3[key][:,channel]>threshold)
        if len(where):
            i=int(where[0]);indices.update(range(max(0,i-2),min(i+3,len(v3[key]))))
            crossings.append(dict(key=key,channel=channel,threshold=threshold,first_frame=i,frames_above=len(where)))
rows=[]
for i in sorted(indices):
    rows.append(dict(frame=i,hand_head_error=v3['original_task_error'][i].tolist(),foot_error=v3['foot_position_error'][i].tolist(),
                     root_height_delta_v2=float(v3['qpos'][i,2]-v2['qpos'][i,2]),
                     leg_q_delta_v2=(v3['qpos'][i,7:19]-v2['qpos'][i,7:19]).tolist(),
                     arm_q_delta_v2=(v3['qpos'][i,20:]-v2['qpos'][i,20:]).tolist(),
                     leg_success=bool(v3['leg_solver_success'][i]),arm_success=v3['arm_solver_success'][i].tolist()))
output=dict(crossings=crossings,samples=rows)
(BASE/'intent_retarget_v3_pico_failure_probe.json').write_text(json.dumps(output,indent=2))
print(json.dumps(crossings,indent=2))
print(json.dumps([r for r in rows if r['frame'] in {c['first_frame'] for c in crossings}],indent=2))
