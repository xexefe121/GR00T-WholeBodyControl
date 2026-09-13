"""Fixed physical trials; both learned arms are judged with received history."""
import json,time
from pathlib import Path
import numpy as np
import torch
from gear_sonic.envs.mjlab.g1_true23_matched_memory_dynamics import MatchedMemoryNative23Env


def fixed_rows(pool,count):
    return np.asarray(pool)[np.linspace(0,len(pool)-1,count).astype(int)]


def quiet_reference_rows(env,rows):
    rows=np.asarray(rows)
    indices=env.memory['expert_rows'][rows,:2].astype(int)
    clip=torch.as_tensor(indices[:,0]);frame=torch.minimum(torch.as_tensor(indices[:,1]),env.lengths[clip]-1)
    eligible=env.loco_alpha[clip,frame].abs()<=1e-7
    for key in ('joint_velocity','root_velocity','root_omega','feet_velocity','task_velocity','task_omega'):
        values=env.references[key][clip,frame].flatten(1)
        eligible &= values.abs().amax(1)<=.005
    chosen=rows[eligible.numpy()]
    if not len(chosen):raise ValueError('No actual quiet states with a quiet received standing goal')
    return chosen


def hold_received_pose(env,controls):
    """Append one held-pose stream per world, retaining its actual received prefix.

    The first38 slots reproduce the prefix already received at the physical
    start. Subsequent packets repeat the current pose; their derivatives are
    zero. Physical state and controller memory are untouched. Evaluation only.
    """
    keys=('references','lengths','totals','source_starts','source_stops','loco_alpha','standing_legs')
    before={key:getattr(env,key) for key in keys}
    clips=env.clips.clone();frames=env.frames.clone();n=env.count
    length=next(iter(env.references.values())).shape[1]
    if length<controls+38:raise ValueError('reference allocation too short for continuous standing hold')
    prefix=(frames[:,None]+torch.arange(-37,1)[None]).clamp_min(0)
    prefix=torch.minimum(prefix,env.lengths[clips,None]-1)
    env.references={}
    for key,values in before['references'].items():
        selected=values[clips[:,None],prefix]
        held=selected[:,-1:].expand(n,length,*selected.shape[2:]).clone()
        held[:,:38]=selected
        if key.endswith('velocity') or key.endswith('omega'):held[:,38:]=0
        env.references[key]=torch.cat((values,held))
    alpha=before['loco_alpha'][clips,torch.minimum(frames,before['lengths'][clips]-1)]
    if torch.max(abs(alpha))>1e-7:raise ValueError('standing trials require existing closed standing references')
    held_alpha=alpha[:,None].expand(n,length).clone()
    held_alpha[:,:38]=before['loco_alpha'][clips[:,None],prefix]
    env.loco_alpha=torch.cat((before['loco_alpha'],held_alpha))
    count=len(before['lengths'])
    env.lengths=torch.cat((before['lengths'],torch.full((n,),length)))
    env.totals=torch.cat((before['totals'],torch.full((n,),controls+38)))
    env.source_starts=torch.cat((before['source_starts'],torch.full((n,),10**9)))
    env.source_stops=torch.cat((before['source_stops'],torch.full((n,),10**9)))
    env.standing_legs=np.concatenate((before['standing_legs'],before['standing_legs'][clips.numpy()]))
    env.clips[:]=torch.arange(count,count+n)
    env.time_offset+=(frames.numpy()-37)*.02
    env.frames[:]=37;env.episode_end_frames[:]=37+controls
    return before


def evaluate(actor,path,output,training_env,*,memory_mode='replayed',standing_controls=1500,only_standing=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    fw=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
    env=MatchedMemoryNative23Env(root/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',training_env.bank,
        fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml',count=64,device='cpu',canonical_starts=True,
        canonical_worlds=4,physics_backend='mjbatch',memory_bank=training_env.memory_bank,reset_memory=memory_mode,
        preview_library=fw/'native_preview_backend_v3/libtrue23preview.so',experiment_seed=9901)
    env.terminate_tracking_errors=False;env.capture_evaluation_state=True
    results=[];started=time.monotonic();was_training=actor.training;actor.eval()
    def trial(name,rows,clips,controls,standing=False):
        env.restore_starts(np.arange(64),rows,clips,ends=env.memory['expert_rows'][rows,1].astype(int)+controls,
            durations=np.full(64,controls))
        # Identical delay sequence for every candidate on each fixed trial.
        env.delay_rng=np.random.default_rng(9923)
        held_before=hold_received_pose(env,controls) if standing else None
        alive=np.ones(64,bool);failed=np.zeros(64,bool);records=[[] for _ in range(64)]
        with torch.no_grad():
            for step in range(controls):
                _,_,done,info=env.step(actor.target(actor(env.observe())))
                values=np.column_stack((info['root_error'].numpy(),info['yaw_error'].numpy(),
                    info['foot_errors'].numpy(),info['task_errors'].numpy(),info['leg_error'].numpy()))
                q,v=info['physical_qpos'],info['physical_qvel']
                quiet=np.column_stack((np.linalg.norm(q[:,:2]-info['reference_root'][:,:2],axis=1),
                    np.linalg.norm(v[:,:3],axis=1),np.max(abs(v[:,6:]),axis=1),
                    np.arccos(np.clip(1-2*np.sum(q[:,4:6]**2,axis=1),-1,1))))
                for i in np.flatnonzero(alive):records[i].append(np.r_[values[i],quiet[i]])
                failed |= alive&info['failed'].numpy()
                alive &= ~done.numpy()
                if not alive.any():break
        cases=[]
        for i,record in enumerate(records):
            a=np.asarray(record);p95=np.percentile(a,95,axis=0);leg=float(np.sqrt(np.mean(a[:,7]**2)))
            complete=len(a)==controls and not failed[i]
            tracking=bool(p95[0]<=.2 and p95[1]<=np.deg2rad(15) and max(p95[2:4])<=.12
                and np.all(p95[4:7]<=[.15,.15,.1]) and leg<=.15)
            quiet=bool(p95[8]<=.05 and p95[1]<=np.deg2rad(5) and p95[9]<=.05 and p95[10]<=.5
                and a[:,10].max()<=2 and a[:,11].max()<=.15)
            cases.append(dict(row=int(rows[i]),controls=len(a),physical_complete=bool(complete),tracking=tracking,
                quiet=quiet,passed=bool(complete and (quiet if standing else tracking)),
                root_p95=float(p95[0]),feet_p95=p95[2:4].tolist(),tasks_p95=p95[4:7].tolist(),leg_rmse=leg))
        report=dict(name=name,standing=standing,cases=cases,passed=sum(c['passed'] for c in cases),
            physical_complete=sum(c['physical_complete'] for c in cases),trials=64,requested_seconds=controls*.02)
        results.append(report);(output/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        if held_before is not None:
            for key,value in held_before.items():setattr(env,key,value)
        print(json.dumps(dict(fixed_trial=name,passed=report['passed'],physical_complete=report['physical_complete'])),flush=True)
    try:
        if not only_standing:
            for clip,pools in env.pool_arrays.items():
                rows=fixed_rows(pools['source'],64)
                trial('movement_'+env.meta['clips'][clip]['name'],rows,np.full(64,clip),100)
        initial=fixed_rows(quiet_reference_rows(env,env.initial_pool),32)
        terminal=fixed_rows(quiet_reference_rows(env,np.concatenate([p['stand'] for p in env.pool_arrays.values() if len(p['stand'])])),32)
        rows=np.r_[initial,terminal];trial('standing',rows,env.memory['expert_rows'][rows,0].astype(int),standing_controls,True)
        if not only_standing:
            pools=list(env.pool_arrays.values())
            rows=np.r_[fixed_rows(np.concatenate([p['acquire'] for p in pools]),32),fixed_rows(np.concatenate([p['brake'] for p in pools]),32)]
            trial('transition',rows,env.memory['expert_rows'][rows,0].astype(int),200)
    finally:env.preview.close();actor.train(was_training)
    movement=[r for r in results if r['name'].startswith('movement_')]
    report=dict(kind='fixed_native_physical_trials',evaluation_version=2,standing_reference='received_pose_held_30s',memory_mode=memory_mode,actor=str(path),
        movement_passed=sum(r['passed'] for r in movement),movement_physical=sum(r['physical_complete'] for r in movement),
        standing_passed=next(r['passed'] for r in results if r['name']=='standing'),
        transition_passed=next((r['passed'] for r in results if r['name']=='transition'),0),
        physical_completions=sum(r['physical_complete'] for r in results),
        physical_seconds=sum(sum(c['controls'] for c in r['cases'])*.02 for r in results),
        all_passed=all(r['passed']==64 for r in results),wall_seconds=time.monotonic()-started,
        full_motion_qualification=False,simulation_ready=False,cases=results)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
