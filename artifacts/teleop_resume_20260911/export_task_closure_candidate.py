"""Copy an existing actor with explicit received-task closure semantics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    target=a.output/'actor.onnx';shutil.copy2(a.actor,target)
    with np.load(a.actor.with_suffix('.normalization.npz'),allow_pickle=False) as z:
        payload={key:z[key].copy() for key in z.files}
    payload['task_closure_stand']=np.array(True)
    np.savez(target.with_suffix('.normalization.npz'),**payload)
    request=dict(kind='received_task_closure_standing_candidate',source_actor=str(a.actor),
        actor_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),actor_weights_unchanged=True,
        task_closure_stand=True,future_reference_frames=0,simulation_qualified=False,
        hardware_authorized=False,changes='standing blend uses received task closure instead of redundant arm joint branch')
    (a.output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    print(str(target),flush=True)


if __name__=='__main__':main()
