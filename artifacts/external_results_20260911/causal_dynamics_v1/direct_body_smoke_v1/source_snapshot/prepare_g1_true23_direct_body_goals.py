"""Prepare causal BFM current-pose inputs from native reference kinematics."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive,NEW
from gear_sonic.utils.g1_true23_direct_body_goal import causal_body_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,default=NEW/'causal_dynamics_v1/bank')
    a=p.parse_args();output=a.bank/'bfm_reference_inputs_v1.npz'
    if output.exists():raise FileExistsError(output)
    meta=json.loads((a.bank/'bank.json').read_text())
    meta['clips']+=json.loads((a.bank/'interruptions.json').read_text())['clips']
    model,c,_,_,_=load_case('walk003');data=mujoco.MjData(model)
    body_ids=[model.body(name).id for name in c['body_names']]
    payload={}
    for i,row in enumerate(meta['clips']):
        refs=archive(a.bank/row['file']);n=len(refs['states'])
        positions=np.empty((n,24,3));quaternions=np.empty((n,24,4))
        for j,q in enumerate(refs['states'][:,:30]):
            data.qpos[:]=q;mujoco.mj_kinematics(model,data)
            positions[j]=data.xpos[body_ids];quaternions[j]=data.xquat[body_ids]
        motion=dict(joint_pos=refs['joint'],body_pos_w=positions,body_quat_w=quaternions)
        payload[f'body_{i}']=causal_body_features(motion,c)
        alpha=np.zeros(n,np.float32);blend=0.
        for j in range(n):
            alpha[j]=blend
            near=min(np.max(np.abs(refs['joint'][j]-refs['joint'][0])),np.max(np.abs(refs['joint'][j]-c['default_q'])))<=.04
            moving=not near or np.max(np.abs(refs['joint_velocity'][j]))>.10 or np.linalg.norm(refs['root_velocity'][j])>.025 or np.linalg.norm(refs['root_omega'][j])>.10
            fault=row.get('generated_interruption',False) and j>=row['source_stop']+11
            blend=float(np.clip(blend+np.clip(float(moving and not fault)-blend,-.04,.04),0,1))
        payload[f'alpha_{i}']=alpha
        print(json.dumps(dict(clip=row['name'],frames=n)),flush=True)
    np.savez_compressed(output,**payload)
    print(str(output),flush=True)

if __name__=='__main__':main()
