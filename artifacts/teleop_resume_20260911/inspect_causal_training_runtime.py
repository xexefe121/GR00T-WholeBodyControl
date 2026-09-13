import inspect, json
from pathlib import Path
import torch, mujoco, mujoco_warp
print('runtime', torch.__version__, mujoco.__version__, torch.cuda.is_available(), flush=True)
print('warp_put_data', inspect.signature(mujoco_warp.put_data))
print('warp_put_model', inspect.signature(mujoco_warp.put_model))
root = Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')
p = root/'direct_target_causal_width512_student_v1/fit/student_head.pt'
saved = torch.load(p, map_location='cpu', weights_only=False)
def describe(v):
    if isinstance(v, dict): return {k: describe(x) for k,x in v.items() if k not in ('optimizer_state','optimizer')}
    if isinstance(v, torch.Tensor): return {'shape':list(v.shape),'dtype':str(v.dtype)}
    if isinstance(v, (int,float,str,bool,type(None))): return v
    return str(type(v))
print(json.dumps({k:describe(saved[k]) for k in ('actor_state','feature_mean','feature_std','joint_span')}, indent=2))
import numpy as np
for p in (root/'pico_walk002_labels_v1/collection/pico/labels.npz', root/'pico_walk002_labels_v1/collection/walk002/labels.npz', root/'fast_feedback_walk003_v1/baseline_v1/nominal/trace.npz'):
    with np.load(p) as z: print(p, {k:(z[k].shape,str(z[k].dtype)) for k in z.files})
