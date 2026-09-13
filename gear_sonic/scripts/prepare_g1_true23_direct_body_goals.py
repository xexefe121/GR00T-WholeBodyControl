"""Prepare causal BFM current-pose inputs from native reference kinematics."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive,NEW
from gear_sonic.utils.g1_true23_direct_body_goal import causal_body_features,task_closure_signature,task_closure_standing


def reference_blend(refs,motion,contract,row,task_closure_stand=False):
    """Pre-command alpha and post-command desire from received prefixes only."""
    n=len(refs['joint']);alpha=np.zeros(n,np.float32);desired=np.zeros(n,np.float32);blend=0.
    anchor=None
    if task_closure_stand:
        quats=Rotation.from_matrix(refs['task_rotation'].reshape(-1,3,3)).as_quat()[:,[3,0,1,2]].reshape(n,3,4)
        anchor=task_closure_signature({k:v[0] for k,v in motion.items()},refs['tasks'][0],quats[0],contract)
    for j in range(n):
        alpha[j]=blend
        near=min(np.max(np.abs(refs['joint'][j]-refs['joint'][0])),np.max(np.abs(refs['joint'][j]-contract['default_q'])))<=.04
        moving=not near or np.max(np.abs(refs['joint_velocity'][j]))>.10 or np.linalg.norm(refs['root_velocity'][j])>.025 or np.linalg.norm(refs['root_omega'][j])>.10
        if task_closure_stand:
            signature=task_closure_signature({k:v[j] for k,v in motion.items()},refs['tasks'][j],quats[j],contract)
            closed=task_closure_standing(signature,anchor,joint_velocity=refs['joint_velocity'][j],
                root_velocity=refs['root_velocity'][j],root_omega=refs['root_omega'][j],feet_velocity=refs['feet_velocity'][j],
                task_velocity=refs['task_velocity'][j],task_omega=refs['task_omega'][j])
            moving=not closed
        fault=row.get('generated_interruption',False) and j>=row['source_stop']+11
        desired[j]=float(moving and not fault)
        blend=float(np.clip(blend+np.clip(desired[j]-blend,-.04,.04),0,1))
    return alpha,desired


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,default=NEW/'causal_dynamics_v1/bank')
    p.add_argument('--task-closure-stand',action='store_true')
    a=p.parse_args();output=a.bank/'bfm_reference_inputs_v1.npz'
    if output.exists():raise FileExistsError(output)
    meta=json.loads((a.bank/'bank.json').read_text())
    rows=meta['clips']+json.loads((a.bank/'interruptions.json').read_text())['clips']
    model,c,_,_,_=load_case('walk003');data=mujoco.MjData(model)
    body_ids=[model.body(name).id for name in c['body_names']]
    payload={}
    for i,row in enumerate(rows):
        refs=archive(a.bank/row['file']);n=len(refs['states'])
        positions=np.empty((n,24,3));quaternions=np.empty((n,24,4))
        for j,q in enumerate(refs['states'][:,:30]):
            data.qpos[:]=q;mujoco.mj_kinematics(model,data)
            positions[j]=data.xpos[body_ids];quaternions[j]=data.xquat[body_ids]
        motion=dict(joint_pos=refs['joint'],body_pos_w=positions,body_quat_w=quaternions)
        payload[f'body_{i}']=causal_body_features(motion,c)
        alpha,desired=reference_blend(refs,motion,c,row,a.task_closure_stand)
        payload[f'alpha_{i}']=alpha
        payload[f'desired_{i}']=desired
        print(json.dumps(dict(clip=row['name'],frames=n)),flush=True)
    np.savez_compressed(output,task_closure_stand=np.bool_(a.task_closure_stand),**payload)
    meta['task_closure_stand']=bool(a.task_closure_stand)
    meta['body_goal_semantics']='received_task_closure_v1' if a.task_closure_stand else 'received_joint_nearness_v1'
    (a.bank/'bank.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(str(output),flush=True)

if __name__=='__main__':main()
