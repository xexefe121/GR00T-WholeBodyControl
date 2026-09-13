"""Add simulation-only interrupted reference variants; preserve original bank."""
import argparse,json
from pathlib import Path
import numpy as np
from gear_sonic.scripts.prepare_g1_true23_causal_dynamics import BUNDLE,OLD,archive
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,motion_states
from gear_sonic.utils.g1_true23_received_features import prepare_reference
from gear_sonic.utils.g1_true23_causal_stop import StandingReference


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.bank/'bank.json').read_text());extra=[]
    destination=a.bank/'interruptions';destination.mkdir(exist_ok=False)
    for row in meta['clips']:
        if not row['training']:continue
        clip=row['name'];model,c,_,timeline,_=load_native_bundle(BUNDLE,clip)
        motion=archive(OLD/'mjbatch_intent_floor_inputs_v1'/clip/'reference.npz')
        original=archive(BUNDLE/clip/'original29.npz')
        for fraction in (.2,.65):
            cut=11+row['source_start']+int((row['source_stop']-row['source_start'])*fraction)
            fields={k:v[cut] for k,v in motion.items() if k!='fps'}
            stop=StandingReference(model,fields,timeline['configured_standing_qpos'],meta['tasks'])
            values=[];positions=[];quats=[]
            for _ in range(400):
                f,pos,quat=stop.next();values.append(f);positions.append(pos);quats.append(quat)
            generated={k:np.concatenate((motion[k][:cut+1],np.stack([v[k] for v in values]))) for k in fields}
            generated['fps']=np.array([50.])
            tasks=dict(source_task_position_w=np.concatenate((original['source_task_position_w'][:cut+1],positions)),
                source_task_quaternion_wxyz=np.concatenate((original['source_task_quaternion_wxyz'][:cut+1],quats)))
            ref=prepare_reference(generated,tasks,c);states=motion_states(generated)
            name=f'{clip}_stop{int(fraction*100)}';file=destination/(name+'.npz')
            np.savez_compressed(file,**ref,states=states)
            extra.append(dict(name=name,file='interruptions/'+file.name,training=True,length=len(states),
                total=len(states)-11,initial_frame=10,source_start=row['source_start'],source_stop=cut-10,
                generated_interruption=True,original_source_prefix_preserved=True))
    (a.bank/'interruptions.json').write_text(json.dumps(dict(clips=extra,original_bank_unchanged=True),indent=2)+'\n')
    print(json.dumps(dict(interruption_variants=len(extra))))


if __name__=='__main__':main()
