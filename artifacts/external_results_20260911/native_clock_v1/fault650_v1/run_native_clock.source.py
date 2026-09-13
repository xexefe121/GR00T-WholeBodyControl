"""Prepared feedback with an independent C++ plant and persistent BFM worker.

Simulation only, Linux/WSL. Clock/deadline failures remain failures. Prepared
motion policy is a timing benchmark, not a general live-input controller.
"""
import argparse
import ctypes as ct
import json
import os
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import sys,time,traceback
import shutil
import gc
import numpy as np
import mujoco

ROOT=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
NEW=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')
SOURCE=NEW/'direct_target_width251_evaluation_v2/source_draft_v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SOURCE))
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states
from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed
from runtime_common import BUNDLE,REFERENCE,TEACHER,ONNX,DEPS,archive,finite_json
sys.path.insert(0,str(DEPS))
from evaluate_direct_target_student import source_metrics
from quiet_metrics import standing_windows,quiet_diagnostic
from student_linear_runtime import infer_base
from direct_features import DirectFeatures
sys.path.insert(0,str(NEW/'direct_target_width251_collection_v1/source_draft_v1'))
from collection_math import difference_function,committed_target

DOUBLE=ct.POINTER(ct.c_double)
INT64=ct.POINTER(ct.c_int64)


class CompactBackward:
    """Encode one row when every received standing row is exactly identical."""
    def __init__(self,session):self.session=session;self.duplicates=0
    def run(self,names,feed):
        if len(feed['state'])>1 and all(np.array_equal(v,np.broadcast_to(v[:1],v.shape)) for v in feed.values()):
            feed={k:np.ascontiguousarray(v[:1]) for k,v in feed.items()};self.duplicates+=1
        return self.session.run(names,feed)


def ptr(array):return array.ctypes.data_as(DOUBLE)


def benchmark_pass(result):
    m=result['source_metrics']
    tracking=(m['source_controls']==m['requested_source_controls'] and
        m['original_root_world_p95_m']<=.20 and m['original_root_yaw_abs_p95_deg']<=15 and
        np.all(np.asarray(m['original_hand_head_relative_p95_m'])<=[.15,.15,.10]) and
        np.all(np.asarray(m['world_axis_relative_foot_p95_m'])<=.12) and m['leg_rmse_rad']<=.15)
    return bool(result['physical_complete'] and tracking and result['main_quiet'] and
        result['main_quiet']['quiet_standing_diagnostic_pass'] and
        result['quiet']['quiet_standing_diagnostic_pass'] and
        result['controller_deadline_misses']==0 and result['plant_steps_over2ms_late']==0 and
        result['missed_observation_publications']==0 and result['first_latched_fault_control']==-1)


class Bridge:
    def __init__(self,libpath,name=None):
        self.lib=ct.CDLL(str(libpath));l=self.lib
        l.clock_shared_size.restype=ct.c_size_t
        l.clock_initialize.argtypes=[ct.c_void_p];l.clock_stop.argtypes=[ct.c_void_p]
        l.clock_latch_fault.argtypes=[ct.c_void_p]
        l.clock_read_observation.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),INT64,DOUBLE]
        l.clock_publish_observation.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int64,DOUBLE]
        l.clock_read_result.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_int),INT64,INT64,DOUBLE]
        l.clock_publish_result.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int64,ct.c_int64,DOUBLE]
        l.clock_run.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int64]+[DOUBLE]*10+[INT64,DOUBLE]
        l.clock_run.restype=ct.c_int
        self.shm=shared_memory.SharedMemory(name=name) if name else shared_memory.SharedMemory(create=True,size=l.clock_shared_size())
        self.buffer=(ct.c_char*self.shm.size).from_buffer(self.shm.buf)
        self.address=ct.addressof(self.buffer)
        if not name:l.clock_initialize(self.address)

    def observation(self):
        ident,fault,published=ct.c_int(),ct.c_int(),ct.c_int64();values=np.empty(382)
        if self.lib.clock_read_observation(self.address,ct.byref(ident),ct.byref(fault),ct.byref(published),ptr(values)):
            return ident.value,fault.value,published.value,values
        return None

    def result(self,standing):
        ident,start,finish=ct.c_int(),ct.c_int64(),ct.c_int64();target=np.empty(23)
        if self.lib.clock_read_result(self.address,standing,ct.byref(ident),ct.byref(start),ct.byref(finish),ptr(target)):
            return ident.value,start.value,finish.value,target
        return None

    def close(self,owner=False):
        del self.buffer
        self.shm.close()
        if owner:self.shm.unlink()


def stationary_goal(seed,qpos,anchor,qvel=None):
    # Neutral pose's backward encoding is yaw/translation invariant. Add a
    # bounded measured-frame velocity command to recover the latched anchor.
    states=seed.state[:8].copy();priv=seed.privileged[:8].copy()
    w,x,y,z=qpos[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    dy=np.arctan2(np.sin(anchor[2]-yaw),np.cos(anchor[2]-yaw))
    omega=np.array([0.,0.,np.clip(4*dy,-.8,.8)])
    delta=np.r_[anchor[:2]-qpos[:2],0.]
    if qvel is not None:delta[:2]=4*delta[:2]-qvel[:2]
    delta*=min(1.,.6/max(np.linalg.norm(delta),1e-8))
    c,s=np.cos(yaw),np.sin(yaw);heading=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    for t in range(8):
        positions=np.vstack((np.zeros(3),priv[t,1:73].reshape(24,3)))
        priv[t,223:298]+=(heading.T@delta+np.cross(omega,positions)).reshape(-1)
        priv[t,298:373]+=np.tile(omega,25);states[t,-3:]+=omega
    latent=seed.sessions['backward'].run(None,dict(state=np.ascontiguousarray(states,np.float32),privileged=np.ascontiguousarray(priv,np.float32)))[0].mean(0,keepdims=True)
    return 16*latent/np.maximum(np.linalg.norm(latent,axis=-1,keepdims=True),1e-12)


def terminal_braking_goal(seed,frame,qpos,qvel):
    from gear_sonic.utils.g1_true23_bfm_seed_observations import _quaternion_matrix
    frame=min(frame,len(seed.state)-1);stop=min(frame+8,len(seed.state))
    states=seed.state[frame:stop].copy();priv=seed.privileged[frame:stop].copy()
    actrot=_quaternion_matrix(qpos[3:7]);ay=np.arctan2(actrot[1,0],actrot[0,0])
    refrot=_quaternion_matrix(seed.motion['body_quat_w'][frame,0]);ry=np.arctan2(refrot[1,0],refrot[0,0])
    dy=np.arctan2(np.sin(ry-ay),np.cos(ry-ay));omega=np.array([0.,0.,np.clip(4*dy,-.8,.8)])
    delta=4*(seed.motion['body_pos_w'][frame,0]-qpos[:3])-qvel[:3]
    delta[2]=0;delta*=min(1.,.6/max(np.linalg.norm(delta),1e-8))
    heading_actual=np.array([[np.cos(ay),-np.sin(ay),0],[np.sin(ay),np.cos(ay),0],[0,0,1.]])
    for t,index in enumerate(range(frame,stop)):
        rot=_quaternion_matrix(seed.motion['body_quat_w'][index,0]);yaw=np.arctan2(rot[1,0],rot[0,0])
        heading=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
        positions=np.vstack((np.zeros(3),priv[t,1:73].reshape(24,3)));velocity=seed.motion['body_lin_vel_w'][index,0]
        local=heading_actual.T@(velocity+delta)-heading.T@velocity
        priv[t,223:298]+=(local+np.cross(omega,positions)).reshape(-1)
        priv[t,298:373]+=np.tile(omega,25);states[t,-3:]+=omega
    latent=seed.sessions['backward'].run(None,dict(state=np.ascontiguousarray(states,np.float32),privileged=np.ascontiguousarray(priv,np.float32)))[0].mean(0,keepdims=True)
    return 16*latent/np.maximum(np.linalg.norm(latent,axis=-1,keepdims=True),1e-12)


def worker(libpath,name,standing,ready,stop,out,optimize,braking,fault_control):
    bridge=None;rows=[];fault_anchor=None
    try:
        bridge=Bridge(libpath,name)
        if optimize and hasattr(os,'sched_setaffinity'):
            available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{available[min(2 if standing else 4,len(available)-1)]})
        model,c,original,timeline,_=load_native_bundle(BUNDLE,'walk003')
        for k in ('default_q','kp','kd','training_effort','joint_limits'):c[k]=np.asarray(c[k])
        if standing:
            seed=Native23BFMRolloutSeed(model,c,original,ONNX,dependency_directory=DEPS,threads=1)
            if optimize:seed.sessions['backward']=CompactBackward(seed.sessions['backward'])
        else:
            import onnxruntime as ort
            from gear_sonic.utils.g1_true23_mjbatch_mpc import load_motion_override
            model,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
            motion,_=load_motion_override(REFERENCE,BUNDLE,'walk003',model,c,original,timeline,manifest)
            goals=DirectFeatures(motion,archive(BUNDLE/'walk003/original29.npz'),c)
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            head=ort.InferenceSession(str(NEW/'direct_target_causal_width512_student_v1/fit/student_head.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
            package=archive(NEW/'fast_feedback_walk003_v1/controller.npz')
            core=NEW/'direct_target_width512_expert_recovery_v2/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py'
            difference=difference_function(core)
        warm_calls=0
        if optimize:
            observation=bridge.observation()
            if observation is not None:
                values=observation[3];q,v=values[:30],values[30:59]
                for _ in range(10):
                    if standing:
                        infer_base(seed,q,v,values[59:82].astype(np.float32),values[82:].astype(np.float32),11,False)
                    else:
                        x=np.r_[goals(q,v,261),values[59:].astype(np.float32)].astype(np.float32)
                        head.run(None,{'features':x[None]})
                    warm_calls+=1
            gc.collect();gc.disable()
        (out/('standing_setup.json' if standing else 'active_setup.json')).write_text(json.dumps(dict(
            noncontrol_warmup_calls=warm_calls,cyclic_gc_enabled_in_loop=gc.isenabled(),
            physical_steps_during_warmup=0,history_commits_during_warmup=0)))
        last=-1;anchor=None;ready.set()
        while not stop.is_set():
            observation=bridge.observation()
            if observation is None or observation[0]<=last:
                time.sleep(.0001);continue
            control,fault,published,values=observation;last=control
            if control<0:continue
            if standing and fault_control is not None and control>=fault_control:
                bridge.lib.clock_latch_fault(bridge.address);fault=1
            if not standing and (control<250 or control>=1269 or fault):continue
            start=time.monotonic_ns();q,v=values[:30],values[30:59]
            if standing:
                previous,history=values[59:82].astype(np.float32),values[82:].astype(np.float32)
                if not fault and control>=1269 and braking:
                    goal=terminal_braking_goal(seed,control+11,q,v)
                    sensed,_=seed._terms(q,v,previous)
                    raw=seed.sessions['actor'].run(None,dict(state=sensed[None],last_action=previous[None],history=history[None],z=goal))[0][0]*5
                    target=c['default_q']+raw*.25*c['training_effort']/c['kp']
                elif not fault and (control<250 or control>=1269):
                    raw,target,_=infer_base(seed,q,v,previous,history,control+11,control>=1269)
                else:
                    if fault and fault_anchor is None:anchor=None
                    if anchor is None or not fault:
                        w,x,y,z=q[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));anchor=np.r_[q[:2],yaw]
                    if fault and fault_anchor is None:fault_anchor=anchor.copy()
                    goal=stationary_goal(seed,q,anchor,v if braking else None)
                    sensed,_=seed._terms(q,v,previous)
                    raw=seed.sessions['actor'].run(None,dict(state=sensed[None],last_action=previous[None],history=history[None],z=goal))[0][0]*5
                    target=c['default_q']+raw*.25*c['training_effort']/c['kp']
                target=np.clip(target,c['joint_limits'][:,0],c['joint_limits'][:,1])
            elif control==250:
                features=np.r_[goals(q,v,control+11),values[59:].astype(np.float32)].astype(np.float32)
                normalized=head.run(None,{'features':features[None]})[0][0]
                limits=np.asarray(c['joint_limits']);span=(limits[:,1]-limits[:,0]).astype(np.float32)
                target=np.clip(np.asarray(c['default_q'])+span.astype(float)*normalized.astype(float),limits[:,0],limits[:,1])
            else:
                i=control-251
                target,_,_,_=committed_target(difference,package['nominal_states'][i],package['feedforward_targets'][i],package['feedback_gains'][i],q,v,np.asarray(c['joint_limits']))
            target=np.ascontiguousarray(target,dtype=np.float64);finish=time.monotonic_ns()
            accepted=bridge.lib.clock_publish_result(bridge.address,int(standing),control,start,finish,ptr(target))
            rows.append([control,published,start,finish,accepted])
    except BaseException:
        (out/('standing_error.txt' if standing else 'active_error.txt')).write_text(traceback.format_exc())
        ready.set()
    finally:
        np.save(out/('standing_calls.npy' if standing else 'active_calls.npy'),np.asarray(rows,np.int64).reshape(-1,5))
        if standing and fault_anchor is not None:(out/'fault_anchor.json').write_text(json.dumps(fault_anchor.tolist()))
        if bridge:bridge.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--controls',type=int,default=3069)
    p.add_argument('--optimize',action='store_true');p.add_argument('--braking',action='store_true')
    p.add_argument('--realtime-priority',action='store_true')
    p.add_argument('--initial-velocity',type=float,default=0.)
    p.add_argument('--fault-control',type=int);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,a.output/'run_native_clock.source.py')
    shutil.copy2(ROOT/'gear_sonic/native/true23_clock.cpp',a.output/'true23_clock.source.cpp')
    shutil.copy2(NEW/'native_clock_v1/libtrue23clock.so',a.output/'libtrue23clock.so')
    assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    model,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,_=load_motion_override(REFERENCE,BUNDLE,'walk003',model,c,original,timeline,manifest)
    data=mujoco.MjData(model);initial=motion_states(motion)[10].copy()
    initial[30]+=a.initial_velocity
    data.qpos[:]=initial[:30];data.qvel[:]=initial[30:];mujoco.mj_forward(model,data)
    libpath=NEW/'native_clock_v1/libtrue23clock.so';bridge=Bridge(libpath)
    values=np.r_[initial,np.zeros(323)]
    bridge.lib.clock_publish_observation(bridge.address,0,0,time.monotonic_ns(),ptr(values))
    context=mp.get_context('fork');stop=context.Event();workers=[]
    try:
        for standing in (False,True):
            ready=context.Event();process=context.Process(target=worker,args=(libpath,bridge.shm.name,standing,ready,stop,a.output,a.optimize,a.braking,a.fault_control))
            process.start();workers.append((process,ready))
        for process,ready in workers:
            if not ready.wait(45):raise TimeoutError('worker setup timeout')
        deadline=time.monotonic()+30
        while True:
            initial_result=bridge.result(True)
            if initial_result and initial_result[0]==0:break
            if time.monotonic()>deadline:raise TimeoutError('standing initialization timeout')
            time.sleep(.01)
        target=initial_result[3]
        if a.optimize and hasattr(os,'sched_setaffinity'):
            os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        scheduling=dict(requested=a.realtime_priority,applied=False)
        if a.realtime_priority:
            try:os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(10));scheduling['applied']=True
            except OSError as error:scheduling['error']=str(error)
        # Exercise active process and steady BFM before choosing clock epoch;
        # no simulation step or extra committed history is used for warmup.
        states=np.empty((a.controls*10+1,60));targets=np.empty((a.controls*10,23));torques=np.empty_like(targets)
        timing=np.empty((a.controls*10,7),np.int64);summary=np.empty(10)
        arrays=[np.ascontiguousarray(c[k],dtype=np.float64) for k in ('kp','kd','native_effort','native_velocity','default_q','training_effort')]
        epoch=time.monotonic_ns()+100_000_000
        started=time.monotonic()
        try:
            count=bridge.lib.clock_run(bridge.address,model._address,data._address,a.controls,epoch,
                *[ptr(x) for x in arrays],ptr(target),ptr(states),ptr(targets),ptr(torques),timing.ctypes.data_as(INT64),ptr(summary))
        finally:
            if scheduling['applied']:
                os.sched_setscheduler(0,os.SCHED_OTHER,os.sched_param(0))
        elapsed=time.monotonic()-started
        np.savez_compressed(a.output/'trace.npz',states=states[:count+1],targets=targets[:count],torques=torques[:count],timing=timing[:count],epoch_ns=np.asarray(epoch))
        complete=count==a.controls*10 and not summary[3]
        controls=count//10;indices=np.arange(controls)*10+10
        trace=dict(qpos=states[np.r_[0,indices],:30],qvel=states[np.r_[0,indices],30:59],
            source_frame=np.minimum(np.arange(controls)+11,len(motion['joint_pos'])-1),global_control=np.arange(controls),physics_substeps=np.full(controls,10),
            physics_qpos=states[:count+1,:30],physics_qvel=states[:count+1,30:59],physics_time=states[:count+1,59],physics_torque=torques[:count])
        audit=json.loads((TEACHER/'recorded_source_audit_v2.json').read_text());original29=archive(BUNDLE/'walk003/original29.npz')
        tracking=source_metrics(model,trace,motion,original29,audit,timeline)
        quiet=quiet_diagnostic(standing_windows(trace,motion,original,original29),not summary[3]) if count>=1500 else None
        main_quiet=None
        if controls>=1569:
            main_trace={k:(v[:15691] if k in ('physics_qpos','physics_qvel','physics_time') else v[:15690] if k=='physics_torque'
                else v[:1570] if k in ('qpos','qvel') else v[:1569]) for k,v in trace.items()}
            main_quiet=quiet_diagnostic(standing_windows(main_trace,motion,original,original29),not summary[3])
        result=dict(physical_complete=bool(complete),requested_controls=a.controls,completed_physics_steps=count,
            controller_deadline_misses=int(summary[1]),plant_steps_over2ms_late=int(summary[2]),physical_failure=bool(summary[3]),
            first_latched_fault_control=int(summary[4]),maximum_speed_ratio=summary[5],maximum_range_excess=summary[6],maximum_effort_ratio=summary[7],
            maximum_plant_wake_lateness_ms=summary[8],missed_observation_publications=int(summary[9]),
            source_metrics=tracking,quiet=quiet,elapsed_s=elapsed,independent_physics_500hz=True,controller_hz=50,
            persistent_standing_worker=True,simulation_time_frozen_for_inference=False,prepared_motion_specific=True,
            affinity_and_duplicate_standing_row_optimization=a.optimize,
            terminal_position_gain=4 if a.braking else 1,terminal_velocity_damping=1 if a.braking else 0,
            process_scheduling=scheduling,main_quiet=main_quiet,
            initial_root_x_velocity_offset=a.initial_velocity,
            injected_fault_control=a.fault_control,packet_loss_end_to_end_test=False,
            general_live_teleop_qualified=False,hardware_authorized=False)
        result['prepared_benchmark_pass']=benchmark_pass(result)
        (a.output/'report.json').write_text(json.dumps(finite_json(result),indent=2)+'\n');print(json.dumps(finite_json(result)),flush=True)
    finally:
        stop.set();bridge.lib.clock_stop(bridge.address)
        for process,_ in workers:
            process.join(10)
            if process.is_alive():process.terminate();process.join(5)
        bridge.close(owner=True)


if __name__=='__main__':main()
