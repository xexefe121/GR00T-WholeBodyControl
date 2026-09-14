"""Bounded offline one-joint feasibility study in full coupled native23 dynamics."""
import argparse,copy,csv,json,time
from pathlib import Path
import mujoco
import numpy as np
from diagnose_pico_preview_failure import restore_physics,set_pd,load_case
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import assess
from gear_sonic.utils.g1_true23_native_preview_guard import NativePreviewGuard
from gear_sonic.utils.g1_true23_sim_preview import json_safe

JOINT='right_shoulder_roll_joint'


def joint_index(model,contract,name=JOINT):
    joint=model.joint(name)
    index=int(joint.qposadr[0])-7
    if not 0<=index<23 or int(joint.dofadr[0])-6!=index:
        raise ValueError('Unexpected native joint mapping')
    np.testing.assert_array_equal(model.jnt_range[joint.id],contract['joint_limits'][index])
    return index


def rollout(model,c,state,previous,target,delay,steps=16,reserve=1e-4):
    d=restore_physics(model,state);expected=d.time
    low=np.full(23,np.inf);high=low.copy();velocity_margin=low.copy()
    requested=[];saturated=[];first=None;saturation_steps=0;peak_effort=0.
    for step in range(steps):
        applied=previous if step<delay else target
        raw=set_pd(d,applied,c['kp'],c['kd'],c['native_effort'],c['joint_limits'])
        requested.append(raw.copy());saturated.append(d.ctrl.copy())
        saturation_steps+=int(np.any(abs(raw)>c['native_effort']))
        mujoco.mj_step(model,d);expected+=.002
        low=np.minimum(low,d.qpos[7:]-c['joint_limits'][:,0])
        high=np.minimum(high,c['joint_limits'][:,1]-d.qpos[7:])
        velocity_margin=np.minimum(velocity_margin,c['native_velocity']-abs(d.qvel[6:]))
        reasons,values=assess(d,c,expected)
        peak_effort=max(peak_effort,float(np.max(abs(d.qfrc_actuator[6:])/c['native_effort'])))
        margins=np.minimum(d.qpos[7:]-c['joint_limits'][:,0],c['joint_limits'][:,1]-d.qpos[7:])
        if np.min(margins)<reserve:reasons=reasons+['preview_reserve']
        if reasons and first is None:
            limiting=int(np.argmin(margins))
            first=dict(substep=step+1,seconds=(step+1)*.002,reasons=reasons,
                minimum_margin_joint=model.joint(limiting+1).name,minimum_position_margin_rad=float(margins[limiting]))
        if 'nonfinite' in reasons or 'engine_warning' in reasons:break
    minimum=float(min(low.min(),high.min())) if steps else float(np.min(np.minimum(
        d.qpos[7:]-c['joint_limits'][:,0],c['joint_limits'][:,1]-d.qpos[7:])))
    result=dict(delay_substeps=delay,horizon_substeps=steps,passed=first is None,
        minimum_position_margin_rad=minimum,minimum_reserve_margin_rad=minimum-reserve,
        minimum_velocity_margin_rad_s=float(velocity_margin.min()) if steps else float(np.min(c['native_velocity']-abs(d.qvel[6:]))),
        minimum_lower_margin_rad=low if steps else d.qpos[7:]-c['joint_limits'][:,0],
        minimum_upper_margin_rad=high if steps else c['joint_limits'][:,1]-d.qpos[7:],
        maximum_applied_effort_ratio=peak_effort,saturated_substeps=saturation_steps,first_failure=first)
    return result,np.asarray(requested),np.asarray(saturated)


def seeds(low,high,q,v,kp,kd,effort,request,previous,candidate):
    regular=np.linspace(low,high,257)
    special=np.array([q,request,previous,candidate,q+(effort+kd*v)/kp,q+(-effort+kd*v)/kp])
    return np.unique(np.r_[regular,np.clip(special,low,high)])


def search_boundary(model,c,state,previous,diagnostic,control,output,library):
    output.mkdir(exist_ok=False);index=joint_index(model,c)
    seed_data=restore_physics(model,state);q=seed_data.qpos[7+index];v=seed_data.qvel[6+index]
    request=np.asarray(diagnostic['requested_target'],float);guard_candidate=np.asarray(diagnostic['candidate_target'],float)
    np.testing.assert_array_equal(seed_data.qpos,diagnostic['qpos'])
    np.testing.assert_array_equal(seed_data.qvel,diagnostic['qvel'])
    runtime_previous=np.asarray(diagnostic['previous_target'],float)
    previous_history_error=float(np.max(abs(previous-runtime_previous)))
    # Pre-replacement trajectories cannot depend on a replacement command.
    prefixes=[]
    for delay in range(7):
        result,_,_=rollout(model,c,state,previous,previous,delay,steps=delay)
        prefixes.append(result)
    guard=NativePreviewGuard(model,c,steps=10,delay_substeps=6,library=library)
    if guard.native is not None:guard.native.reset(seed_data.qpos,seed_data.qvel,runtime_previous)
    else:raise ValueError('Use pinned native preview library for this study')
    values=seeds(*c['joint_limits'][index],q,v,c['kp'][index],c['kd'][index],c['native_effort'][index],
                 request[index],previous[index],guard_candidate[index])
    table=[];details=[];requests=[];applied=[];seen=set();spacing=float(np.diff(np.linspace(*c['joint_limits'][index],257))[0])
    sampling=[]
    for round_index in range(3):
        fresh=[float(x) for x in values if float(x) not in seen];sampling.append(dict(round=round_index,candidates=len(fresh),spacing_rad=spacing))
        for value in fresh:
            seen.add(value);target=request.copy();target[index]=value
            low,high=guard.native.preview(target)
            cold_ok=bool(max(low.max(),high.max())<=0)
            rows=[];candidate_id=len(details)
            for delay in range(7):
                r,raw,sat=rollout(model,c,state,previous,target,delay)
                rows.append(r);requests.append(raw);applied.append(sat)
                failure=r['first_failure']
                table.append(dict(candidate_id=candidate_id,target_rad=value,delay_substeps=delay,
                    passed=r['passed'],minimum_position_margin_rad=r['minimum_position_margin_rad'],
                    minimum_reserve_margin_rad=r['minimum_reserve_margin_rad'],
                    minimum_velocity_margin_rad_s=r['minimum_velocity_margin_rad_s'],
                    maximum_applied_effort_ratio=r['maximum_applied_effort_ratio'],
                    saturated_substeps=r['saturated_substeps'],
                    first_failure_substep=None if failure is None else failure['substep'],
                    first_failure=None if failure is None else ','.join(failure['reasons']),
                    limiting_joint=None if failure is None else failure['minimum_margin_joint'],
                    runtime_fixed_candidate_preview_passed=cold_ok))
            details.append(dict(candidate_id=candidate_id,target_rad=value,round=round_index,
                all_delays_passed=all(r['passed'] for r in rows),
                worst_reserve_margin_rad=min(r['minimum_reserve_margin_rad'] for r in rows),
                runtime_fixed_candidate_preview_passed=cold_ok,
                runtime_fixed_candidate_excess_rad=float(max(low.max(),high.max())),delays=rows))
        best=sorted(details,key=lambda r:(-r['worst_reserve_margin_rad'],abs(r['target_rad']-request[index])))[:3]
        values=np.unique(np.clip(np.concatenate([np.linspace(r['target_rad']-spacing,r['target_rad']+spacing,17) for r in best]),*c['joint_limits'][index]))
        spacing/=8
        print(json.dumps(dict(control=control,round=round_index,candidates=len(details),
            all_delay_survivors=sum(r['all_delays_passed'] for r in details))),flush=True)
    with (output/'target_by_delay.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    np.savez_compressed(output/'effort_traces.npz',requested_effort=np.asarray(requests),saturated_effort=np.asarray(applied),
        candidate_id=np.asarray([r['candidate_id'] for r in table]),delay_substeps=np.asarray([r['delay_substeps'] for r in table]))
    survivors=[r for r in details if r['all_delays_passed'] and r['runtime_fixed_candidate_preview_passed']]
    selected=sorted(survivors,key=lambda r:(-r['worst_reserve_margin_rad'],abs(r['target_rad']-request[index])))[:3]
    result=dict(control=control,joint_name=JOINT,joint_index=index,limits_rad=c['joint_limits'][index],
        previous_applied_target=previous,runtime_previous_target=runtime_previous,
        previous_history_maximum_difference_rad=previous_history_error,
        delay_prefixes=prefixes,sampling=sampling,candidate_count=len(details),delays=list(range(7)),
        total_horizon_seconds=.032,other_targets='22 actor requested targets fixed',
        runtime_comparison='unchanged native cold-state fixed-candidate check, immediate and maximum delay; not a fresh seven-call search',
        original_runtime_guard_verdict=diagnostic['preview_status'],
        all_delay_survivors=sum(r['all_delays_passed'] for r in details),
        runtime_compatible_survivors=len(survivors),selected_candidates=[{k:v for k,v in r.items() if k!='delays'} for r in selected],
        offline_only=True,finite_sampling_is_not_impossibility_proof=True,candidates=details)
    (output/'report.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-run',type=Path,required=True)
    p.add_argument('--reconstruction',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    verified=json.loads((args.reconstruction/'report.json').read_text())
    if not verified['exact_applied_target_replay'] or verified['input_run']!=str(args.input_run):raise ValueError('Matching exact reconstruction required')
    diagnostics={r['control']:r for r in map(json.loads,(args.input_run/'preview_diagnostics.jsonl').read_text().splitlines())}
    report=json.loads((args.input_run/'report.json').read_text());model,c,*_=load_case(report['clip'])
    np.testing.assert_array_equal(c['kp'],report['actual_pd_kp']);np.testing.assert_array_equal(c['kd'],report['actual_pd_kd'])
    controls=sorted(verified['snapshot_boundaries'],reverse=True);results=[]
    for control in controls:
        with np.load(args.reconstruction/f'boundary_{control}.npz') as z:state=z['integration_state'];previous=z['previous_target']
        result=search_boundary(model,c,state,previous,diagnostics[control],control,args.output/f'control_{control}',Path(report['native_preview_library']))
        results.append({k:v for k,v in result.items() if k!='candidates'})
        if control==controls[0] and result['all_delay_survivors']:
            break
    (args.output/'report.json').write_text(json.dumps(json_safe(dict(input_run=str(args.input_run),boundaries=results,
        general_live_teleop_qualified=False)),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
