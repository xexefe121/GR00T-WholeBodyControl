"""Prepared walk003 applied-sequence positive control versus frozen SONIC.

One compiled plant, PD law, recorded application schedule, original goals and
scorer. Offline behavior comparison; no independent-clock timing claim. The
positive control must pass before SONIC is run. No fitting or target correction.
"""
import argparse
import copy
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v))
    Path(path).write_text(json.dumps(value, default=convert, indent=2, allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository-root',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--output-directory',type=Path,required=True)
    a=p.parse_args(); root=a.repository_root; base=a.artifact_root; out=a.output_directory
    out.mkdir(parents=True,exist_ok=False)
    assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    source=base/'sonic23_teleop_resume_20260911/direct_target_width251_evaluation_v2/source_draft_v1'
    sys.path.insert(0,str(source))
    from evaluate_direct_target_student import source_metrics
    from quiet_metrics import standing_windows,quiet_diagnostic
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states
    from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
        CleanTrue23MujocoController, motion_reference_terms,encoder267_from_reference,
        term_major_history,MJ_TO_NATIVE,HARDWARE_23_ACTION_SCALE,
        SAFE_TARGET_DEFAULT_Q_HARDWARE,safe_target_transform_numpy,
    )
    from gear_sonic.utils.g1_23dof_safe_target_transform import (
        SAFE_TARGET_FORMULA,SAFE_TARGET_CONSTANTS_SHA256,SAFE_TARGET_INNER_LOWER_HARDWARE,
        SAFE_TARGET_INNER_UPPER_HARDWARE,
    )
    from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair
    from gear_sonic.utils.g1_true23_sonic_library_replay import ExactHashSonicPolicy
    from gear_sonic.teleop.cpu_paced_inference import nonspinning_options
    bundle=root/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    reference=base/'sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz'
    historical=base/'sonic23_teleop_resume_20260911/native_clock_v1/braking_v5'
    audit_path=base/'sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/recorded_source_audit_v2.json'
    model,c,original,timeline,manifest=load_native_bundle(bundle,'walk003')
    motion,receipt=load_motion_override(reference,bundle,'walk003',model,c,original,timeline,manifest)
    original29=archive(bundle/'walk003/original29.npz'); old=archive(historical/'trace.npz')
    audit=json.loads(audit_path.read_text())
    count=len(old['targets']); controls=count//10
    assert count==30690 and controls==3069
    np.testing.assert_array_equal(motion_states(motion)[10],old['states'][0,:59])
    schedule=old['timing'][:,4].astype(int)
    age=np.arange(count)//10-schedule
    assert np.all((age==0)|(age==1)) and np.all(schedule.reshape(-1,10)[:,-1]==np.arange(controls))
    assert np.all(schedule[:10]==0)
    # Bind each archived command ID to a unique target. No state playback.
    commands=old['targets'].reshape(-1,10,23)[:,-1].copy()
    np.testing.assert_array_equal(commands[schedule],old['targets'])
    gains={k:np.ascontiguousarray(c[k],np.float64) for k in ('kp','kd','native_effort','native_velocity')}
    native_source=root/'gear_sonic/native/true23_matched_step.cpp'
    package=Path(mujoco.__file__).parent; library=next(package.glob('libmujoco.so*'))
    libpath=out/'libmatched_step.so'
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-I'+str(package/'include'),str(native_source),str(library),'-Wl,-rpath,'+str(package),'-o',str(libpath)],check=True)
    lib=ct.CDLL(str(libpath)); ptr=ct.POINTER(ct.c_double)
    lib.matched_step.argtypes=[ct.c_void_p,ct.c_void_p]+[ptr]*5+[ct.c_double,ptr]
    lib.matched_step.restype=ct.c_int
    def dp(x): return x.ctypes.data_as(ptr)
    def physics(d,target,step):
        requested=np.empty(23)
        failure=lib.matched_step(model._address,d._address,dp(np.ascontiguousarray(target,np.float64)),
            *[dp(gains[k]) for k in ('kp','kd','native_effort','native_velocity')],(step+1)*.002,dp(requested))
        return failure,requested
    mujoco.mj_saveModel(model,str(out/'prepared_model.mjb'),None)
    # Save every numeric model array plus scalar solver options, not just XML.
    numeric={k:np.array(getattr(model,k),copy=True) for k in dir(model) if isinstance(getattr(model,k),np.ndarray)}
    np.savez_compressed(out/'prepared_model_arrays.npz',**numeric)
    inputs=[bundle/'manifest.json',bundle/'contract.json',bundle/'prepared_model_arrays.npz',bundle/'native_prepared.xml',
            reference,bundle/'walk003/original29.npz',bundle/'walk003/timeline.json',historical/'trace.npz',historical/'report.json',audit_path,native_source,Path(__file__)]
    binding=dict(kind='matched_walk003_frozen_sonic_v1',inputs_sha256={str(x):sha(x) for x in inputs},
        compiled_model_sha256=sha(out/'prepared_model.mjb'),all_numeric_model_arrays_sha256=sha(out/'prepared_model_arrays.npz'),
        actuation={k:v for k,v in gains.items()},joint_names=c['joint_names'],
        solver_options={k:(getattr(model.opt,k).tolist() if isinstance(getattr(model.opt,k),np.ndarray) else getattr(model.opt,k))
                        for k in dir(model.opt) if not k.startswith('_') and isinstance(getattr(model.opt,k),(np.ndarray,int,float))},
        initialization='historical motion_states(reference)[10], one mj_forward, no added phase or state reset',
        initial_state=old['states'][0],phases=timeline['phases'],extra_hold_s=30,
        schedule='same archived command ID at every 2ms physics step; previous/current command only',
        schedule_age_patterns=np.unique(age.reshape(-1,10),axis=0).tolist(),
        same_physical_plant_object_for_both=True,controller_specific_reference_repair=False,
        positive_control='recorded applied PD targets; no policy history or normalized action exists in this replay',
        sonic_initial_history='10 repeats of measured initial proprioception; previous safe action zero startup sentinel; thereafter own applied target history only',
        sonic_action_decoding=dict(default_hardware=SAFE_TARGET_DEFAULT_Q_HARDWARE,scale_hardware=HARDWARE_23_ACTION_SCALE,
            raw_order='native IsaacLab23',target_order=c['joint_names'],formula=SAFE_TARGET_FORMULA,
            constants_sha256=SAFE_TARGET_CONSTANTS_SHA256,inner_lower=SAFE_TARGET_INNER_LOWER_HARDWARE,inner_upper=SAFE_TARGET_INNER_UPPER_HARDWARE,
            recomputed_from_plant_gains=False),
        independent_timing_requalified=False,offline_recorded_application_schedule=True,braking_filter_enabled=False,
        training_run=False,hardware_authorized=False)
    dump(out/'comparison_contract.json',binding)
    np.save(out/'application_command_id.npy',schedule)
    # Prepared task is common; replace only the hand/head representation with
    # its existing original29 values for SONIC. Positive scoring uses these same
    # original goals. No FK-derived controller-specific substitute is used.
    tail=controls+12-len(motion['joint_pos'])
    extended={k:(np.concatenate((v,np.repeat(v[-1:],tail,axis=0))) if k!='fps' and tail>0 else v.copy()) for k,v in motion.items()}
    ref_packets=[]
    for control in range(controls):
        anchor=control+9; original_index=min(anchor,len(original29['virtual_vr21'])-1)
        packet=motion_reference_terms(extended,anchor)
        vr=original29['virtual_vr21'][original_index]
        packet['vr_3point_local_target']=vr[:9].tolist();packet['vr_3point_local_orn_target']=vr[9:].tolist()
        # Original local goals are expressed in the ORIGINAL pelvis frame.
        # Pairing them with the adapted pelvis orientation rotates the task.
        # Joint retargeting and the historical physical initialization stay as
        # supplied; original root/head/hand goals govern both arms' scoring.
        root_quat=original29['source_qpos29'][original_index,3:7]
        packet['reference_anchor_quaternion_xyzw']=root_quat[[1,2,3,0]].tolist()
        from gear_sonic.utils.g1_true23_clean_mujoco_teleop import _quaternion_matrix
        reconstructed=np.array(vr[:9]).reshape(3,3)@_quaternion_matrix(root_quat).T
        expected=(original29['source_task_position_w'][original_index]-original29['source_qpos29'][original_index,:3])
        np.testing.assert_allclose(reconstructed,expected,atol=2e-6,rtol=0)
        ref_packets.append(packet)
    dump(out/'references.json',ref_packets)
    binding['reference_packets_sha256']=sha(out/'references.json')
    dump(out/'comparison_contract.json',binding)

    def score(arr, failure):
        n=len(arr['targets']); full=n//10
        indices=np.r_[0,np.arange(full)*10+10]
        tr=dict(qpos=arr['states'][indices,:30],qvel=arr['states'][indices,30:59],
            source_frame=np.minimum(np.arange(full)+11,len(motion['joint_pos'])-1),global_control=np.arange(full),
            physics_substeps=np.full(full,10),physics_qpos=arr['states'][:full*10+1,:30],
            physics_qvel=arr['states'][:full*10+1,30:59],physics_time=arr['states'][:full*10+1,59],physics_torque=arr['applied_torque'][:full*10])
        metrics=source_metrics(model,tr,motion,original29,audit,timeline)
        tracking=bool(metrics.get('source_controls')==819 and metrics['original_root_world_p95_m']<=.20 and
            metrics['original_root_yaw_abs_p95_deg']<=15 and np.all(np.array(metrics['original_hand_head_relative_p95_m'])<=[.15,.15,.10]) and
            np.all(np.array(metrics['world_axis_relative_foot_p95_m'])<=.12) and metrics['leg_rmse_rad']<=.15)
        quiet=quiet_diagnostic(standing_windows(tr,motion,original,original29),not failure) if full>=150 else None
        return dict(physical_complete=n==count and not failure,completed_steps=n,simulation_duration_s=n*.002,
            physical_failure_bitmask=failure,source_metrics=metrics,original_tracking_pass=tracking,last_three_seconds_quiet=quiet,
            included_extra_hold_s=max(0,(n-15690)*.002),continuous_30s_quiet_verified=False),tr

    def run(mode, policy=None):
        d=mujoco.MjData(model);d.qpos[:]=old['states'][0,:30];d.qvel[:]=old['states'][0,30:59];mujoco.mj_forward(model,d)
        states=[np.r_[d.qpos,d.qvel,d.time]]; targets=[];requested=[];applied=[];actual=[];contact_rows=[];contacts=[]
        proposals=[];raws=[];encoders=[];histories=[];prior_actions=[];pre_states=[];statuses=[];failure=0
        inference_error=None; actor_times=[]
        sonic=CleanTrue23MujocoController.__new__(CleanTrue23MujocoController)
        sonic.module=mujoco;sonic.model=model;sonic.data=d;sonic.previous_safe_native=np.zeros(23,np.float32)
        sonic.history=[];sonic.buffered_robot_pelvis_q9=d.qpos[3:7].copy()
        for control in range(controls):
            pre_states.append(np.r_[d.qpos,d.qvel,d.time]);packet=ref_packets[control]
            if mode=='sonic':
                # Refresh observation transforms on a copy so observation reads
                # cannot alter the plant integration state relative to replay.
                observation=copy.copy(d);sonic.data=observation
                frame=sonic._policy_frame();sonic.data=d
                sonic.history=[frame.copy() for _ in range(10)] if not sonic.history else [*sonic.history[1:],frame]
                enc=encoder267_from_reference(packet,sonic.buffered_robot_pelvis_q9)
                history=term_major_history(sonic.history);prior_actions.append(sonic.previous_safe_native.copy())
                start=time.perf_counter()
                try: raw,decoder=policy.infer(enc,history); safe,target=safe_target_transform_numpy(raw)
                except (ValueError,RuntimeError) as e: inference_error=str(e);break
                actor_times.append((time.perf_counter()-start)*1000)
                raws.append(raw);encoders.append(enc);histories.append(history)
                proposals.append(target.astype(np.float64))
                buffered=d.qpos[3:7].copy()
            else: proposals.append(commands[control].copy())
            for sub in range(10):
                step=control*10+sub;target=proposals[schedule[step]]
                failure,demand=physics(d,target,step)
                states.append(np.r_[d.qpos,d.qvel,d.time]);targets.append(target.copy());requested.append(demand)
                applied.append(d.ctrl.copy());actual.append(d.qfrc_actuator[6:].copy())
                contact_rows.append(d.ncon)
                if sub==0:
                    contacts.append([dict(geom=d.contact[i].geom.tolist(),distance=float(d.contact[i].dist),
                        efc_address=int(d.contact[i].efc_address)) for i in range(d.ncon)])
                if failure:break
            statuses.append(failure)
            if mode=='sonic':
                sonic.buffered_robot_pelvis_q9=buffered
                # Frozen decoder's applied-action convention, independent of Kp.
                sonic.previous_safe_native=((target-np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE))/np.asarray(HARDWARE_23_ACTION_SCALE))[MJ_TO_NATIVE].astype(np.float32)
            if control%500==0:print(mode,control,'time',d.time,'failure',failure,flush=True)
            if failure:break
        arr=dict(states=np.asarray(states),targets=np.asarray(targets),requested_torque=np.asarray(requested),applied_torque=np.asarray(applied),
            actuator_torque=np.asarray(actual),contact_count=np.asarray(contact_rows),pre_control_state=np.asarray(pre_states),
            proposals=np.asarray(proposals),raw_native=np.asarray(raws),encoder267=np.asarray(encoders),history930=np.asarray(histories),
            previous_safe_native=np.asarray(prior_actions),inference_ms=np.asarray(actor_times),failure_status=np.asarray(statuses))
        report,tr=score(arr,failure)
        report.update(inference_error=inference_error,controller=mode,compiled_model_sha256=binding['compiled_model_sha256'],
            reference_packets_sha256=binding['reference_packets_sha256'],action_semantics_changed=False,
            timing_qualification=False,simulation_ready=False)
        if inference_error:report['physical_complete']=False
        targetdir=out/mode;targetdir.mkdir();np.savez_compressed(targetdir/'trace.npz',**arr)
        dump(targetdir/'contacts_at_first_substep.json',contacts)
        if mode=='positive':
            report['historical_reproduction']={k:dict(exact=np.array_equal(arr[key],old[oldkey][:len(arr[key])]),
                max_abs_difference=float(np.max(abs(arr[key]-old[oldkey][:len(arr[key])]))))
                for k,key,oldkey in [('state','states','states'),('target','targets','targets'),('torque','applied_torque','torques')]}
        # Keep the original last-three-second test distinct from every rolling
        # three-second window ending in the full additional thirty-second hold.
        if report['physical_complete']:
            failed=[]
            for end in range(1569,3070):
                part={k:(v[:end*10+1] if k in ('physics_qpos','physics_qvel','physics_time') else v[:end*10] if k=='physics_torque'
                    else v[:end+1] if k in ('qpos','qvel') else v[:end]) for k,v in tr.items()}
                result=quiet_diagnostic(standing_windows(part,motion,original,original29),True)
                if not result['quiet_standing_diagnostic_pass']:failed.append(dict(control=end,gates=result['gates']))
            report['continuous_30s_quiet_verified']=not failed
            report['rolling_quiet_windows_checked']=1501;report['rolling_quiet_failed_windows']=failed
        report['trace_sha256']=sha(targetdir/'trace.npz');dump(targetdir/'report.json',report)
        return report,arr

    positive,positive_arrays=run('positive')
    if not (positive['physical_complete'] and positive['original_tracking_pass']):
        dump(out/'report.json',dict(decision='positive_control_not_reproduced_for_original_task',positive=positive,sonic_not_run=True))
        print('STOP: positive control failed; SONIC not run',flush=True);return
    pairdir=root/'artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25'
    pair=load_diagnostic_pair(pairdir/'model_25.diagnostic.encoder.json',pairdir/'model_25.diagnostic.decoder.json')
    options=nonspinning_options();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    policy=ExactHashSonicPolicy(encoder_path=Path(pair['encoder']['path']),decoder_path=Path(pair['decoder']['path']),
        expected_encoder_sha256=pair['encoder']['sha256'],expected_decoder_sha256=pair['decoder']['sha256'],session_options=options)
    binding['frozen_sonic_pair']=pair;dump(out/'comparison_contract.json',binding)
    # Verify same model numeric arrays before controller swap.
    for key,value in numeric.items():np.testing.assert_array_equal(getattr(model,key),value,err_msg=key)
    sonic,sonic_arrays=run('sonic',policy)
    for key,value in numeric.items():np.testing.assert_array_equal(getattr(model,key),value,err_msg=key)
    decision='both_pass' if sonic['physical_complete'] and sonic['original_tracking_pass'] else 'positive_passes_frozen_sonic_fails'
    report=dict(decision=decision,positive=positive,sonic=sonic,numeric_model_unchanged_between_arms=True,
        configuration_contract=str(out/'comparison_contract.json'),actuator_crossover_run=False,simulation_ready=False)
    dump(out/'report.json',report)
    print(json.dumps(dict(decision=decision,positive_duration=positive['simulation_duration_s'],sonic_duration=sonic['simulation_duration_s'])),flush=True)


if __name__=='__main__':main()
