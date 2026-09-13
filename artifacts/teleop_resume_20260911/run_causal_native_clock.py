"""One received-only ONNX controller on the independent native plant.

Linux/WSL simulation only. The packet producer and 500 Hz plant have independent
wall clocks. Both legacy result channels receive the same learned command.
The plant owns applied-action history; proposal history is never committed.
"""
import argparse
import ctypes as ct
import gc
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import queue
import shutil
import sys
import time
import traceback
import pickle
import hashlib

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
NEW = Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')
# Pin NumPy/MuJoCo from the native323 interpreter before exposing ancillary
# ONNX/Torch dependencies. Torch is imported by the existing receiver modules;
# all policy inference below is single-thread CPU ONNX.
for dependency in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
                   '/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):
    if Path(dependency).is_dir():
        sys.path.append(dependency)
sys.path.insert(0, str(ROOT))
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import archive, load_case, metrics, quiet
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet
from gear_sonic.utils.g1_true23_causal_controller import CausalController

from gear_sonic.utils.g1_true23_sim_preview import (
    RUNNER_CONTRACT_VERSION, SimulatorPreviewRejected, publish_checked,
    stop_simulator, json_safe, write_diagnostics, terminal_standing)

DOUBLE = ct.POINTER(ct.c_double)
INT64 = ct.POINTER(ct.c_int64)


def ptr(array):
    return array.ctypes.data_as(DOUBLE)


def publish_native_target(bridge, target, control, started, context):
    # Include diagnostic capture/enforcement in measured controller time.
    finished=time.monotonic_ns();context['finished_ns']=finished
    return [bridge.lib.clock_publish_result(bridge.address,channel,control,started,finished,ptr(target))
            for channel in (0,1)]


def pin_run(args):
    from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import BUNDLE,OLD
    paths=[Path(__file__),args.actor,args.library,args.bank/(args.clip+'.npz'),args.bank/'bank.json',
           BUNDLE/'native_prepared.xml',BUNDLE/'prepared_model_arrays.npz',BUNDLE/'contract.json',
           BUNDLE/args.clip/'timeline.json',BUNDLE/args.clip/'original29.npz',
           OLD/'mjbatch_intent_floor_inputs_v1'/args.clip/'reference.npz']
    paths.extend(ROOT/'gear_sonic/utils'/name for name in (
        'g1_true23_native_preview_guard.py','g1_true23_sim_preview.py',
        'g1_true23_native_targets.py','g1_true23_task_commands.py','g1_true23_locomotion_conditioned.py',
        'g1_true23_controller_state.py','g1_true23_causal_controller.py'))
    if args.native_preview_library:paths.append(args.native_preview_library)
    if args.factory_config:paths.append(args.factory_config)
    for suffix in ('.wrapper.json','.reference.json'):
        sidecar=args.actor.with_suffix(suffix)
        if sidecar.exists():paths.append(sidecar)
    files=[]
    for path in paths:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        files.append(dict(path=str(path),sha256=digest.hexdigest(),bytes=path.stat().st_size))
    (args.output/'run_manifest.json').write_text(json.dumps(dict(
        runner_contract_version=RUNNER_CONTRACT_VERSION,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        files=files,initialization='fresh MjData, bank states[10] FP64 q/v, declared velocity delta, mj_forward'),indent=2)+'\n')
    shutil.copy2(__file__,args.output/'run_causal_native_clock.source.py')


def preview_core_affinity(available):
    """Use two physical cores after the plant core, not two SMT siblings."""
    representatives={}
    for cpu in available:
        path=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:key=((path/'physical_package_id').read_text().strip(),(path/'core_id').read_text().strip())
        except OSError:key=('unknown',cpu)
        representatives.setdefault(key,cpu)
    cores=list(representatives.values())
    return set(cores[1:3] or cores[:1])


class Bridge:
    """Existing C ABI for the independent 500 Hz physical plant."""
    def __init__(self, library, name=None):
        self.lib = ct.CDLL(str(library))
        lib = self.lib
        lib.clock_shared_size.restype = ct.c_size_t
        lib.clock_initialize.argtypes = [ct.c_void_p]
        lib.clock_stop.argtypes = [ct.c_void_p]
        lib.clock_latch_fault.argtypes = [ct.c_void_p]
        lib.clock_set_spin_ns.argtypes = [ct.c_int64]
        if hasattr(lib,'clock_wait_observation'):
            lib.clock_wait_observation.argtypes=[ct.c_void_p,ct.c_int,ct.c_int64]
            lib.clock_wait_observation.restype=ct.c_int
        lib.clock_read_observation.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), INT64, DOUBLE]
        lib.clock_publish_observation.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int64, DOUBLE]
        self.observation_width=382
        self.read_observation=lib.clock_read_observation
        self.publish_observation=lib.clock_publish_observation
        if hasattr(lib,'clock_sensor_observation_width'):
            lib.clock_sensor_observation_width.restype=ct.c_int
            self.observation_width=lib.clock_sensor_observation_width()
            if self.observation_width!=462:raise ValueError('Unsupported coherent sensor snapshot layout')
            lib.clock_read_sensor_observation.argtypes=lib.clock_read_observation.argtypes
            lib.clock_publish_sensor_observation.argtypes=lib.clock_publish_observation.argtypes
            self.read_observation=lib.clock_read_sensor_observation
            self.publish_observation=lib.clock_publish_sensor_observation
        lib.clock_read_result.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_int), INT64, INT64, DOUBLE]
        lib.clock_publish_result.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int64, ct.c_int64, DOUBLE]
        lib.clock_run.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int64] + [DOUBLE] * 10 + [INT64, DOUBLE]
        lib.clock_run.restype = ct.c_int
        self.shm = shared_memory.SharedMemory(name=name) if name else shared_memory.SharedMemory(create=True, size=lib.clock_shared_size())
        self.buffer = (ct.c_char * self.shm.size).from_buffer(self.shm.buf)
        self.address = ct.addressof(self.buffer)
        if name is None:
            lib.clock_initialize(self.address)

    def observation(self):
        ident, fault, stamp = ct.c_int(), ct.c_int(), ct.c_int64()
        values = np.empty(self.observation_width)
        if self.read_observation(self.address, ct.byref(ident), ct.byref(fault), ct.byref(stamp), ptr(values)):
            return ident.value, fault.value, stamp.value, values

    def result(self):
        ident, start, finish = ct.c_int(), ct.c_int64(), ct.c_int64()
        target = np.empty(23)
        if self.lib.clock_read_result(self.address, 1, ct.byref(ident), ct.byref(start), ct.byref(finish), ptr(target)):
            return ident.value, target

    def close(self, owner=False):
        del self.buffer
        self.shm.close()
        if owner:
            self.shm.unlink()


def packet_at(motion, original, sequence, standing=False, final=False):
    frame = 11 if standing else sequence
    fields = {k: v[frame] for k, v in motion.items() if k != 'fps'}
    return (Packet(0, sequence, sequence * .02, fields, final),
            original['source_task_position_w'][frame], original['source_task_quaternion_wxyz'][frame])


def import_applied_history(controller, values, control=0):
    """Read the plant's once-per-boundary commit, including held late targets."""
    memory = controller.receiver.history
    memory.prior[:] = values[59:82]
    offset = 82
    for term in memory.terms:
        size = term.size
        term[:] = values[offset:offset + size].reshape(term.shape)
        offset += size
    assert offset == 382
    if hasattr(controller,'import_native_history'):
        controller.import_native_history(control)
    if len(values)==462 and control>0 and hasattr(controller,'previous_native'):
        controller.previous_native=values[439:462].copy()


def initial_sensor_sample(model,data):
    """After mj_forward, all fields describe this one simulated sensor instant."""
    site=model.site('imu_in_pelvis').id
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,data.site_xmat[site])
    return np.r_[data.time,data.qpos[7:],data.qvel[6:],quat,
        data.sensor('imu-pelvis-angular-velocity').data,
        data.sensor('imu-pelvis-linear-acceleration').data]


def producer(packets, epoch_ns, start, stop, motion, original, controls, standing, fault_control, output, realtime=False,library=None):
    rows,stages = [],[]
    try:
        if realtime:os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(40))
        # Packet views/dicts are reference-counted and do not form cycles.
        # Collect the inherited interpreter heap before the clock starts;
        # never pause recorded packet delivery for a full-heap GC traversal.
        gc.collect();gc.disable()
        timer=ct.CDLL(str(library)) if library else None
        if timer and hasattr(timer,'clock_wait_until'):timer.clock_wait_until.argtypes=[ct.c_int64]
        else:timer=None
        start.wait()
        epoch = epoch_ns.value
        last_sequence = controls + 10 if standing else len(motion['joint_pos']) - 1
        for sequence in range(12, last_sequence + 1):
            control = sequence - 11
            if fault_control is not None and control >= fault_control:
                break
            deadline = epoch + control * 20_000_000
            while not stop.is_set():
                remaining = (deadline - time.monotonic_ns()) / 1e9
                if remaining <= 0:
                    break
                if timer:timer.clock_wait_until(deadline)
                else:stop.wait(min(remaining, .02))
            if stop.is_set():
                break
            stamp = time.monotonic_ns()
            item = packet_at(motion, original, sequence, standing, sequence == last_sequence)
            prepared=time.monotonic_ns()
            try:
                packets.put_nowait((stamp, item))
                accepted = 1
            except queue.Full:
                accepted = 0
            rows.append((sequence, deadline, stamp, accepted))
            stages.append((sequence,stamp,prepared,time.monotonic_ns()))
    except BaseException:
        (output / 'producer_error.txt').write_text(traceback.format_exc())
    finally:
        np.save(output / 'producer_calls.npy', np.asarray(rows, np.int64).reshape(-1, 4))
        np.save(output / 'producer_stages.npy',np.asarray(stages,np.int64).reshape(-1,4))


def live_producer(packets,start,stop,url,output,rearm_file=None):
    """Reuse the existing ZMQ body producer; record exact received wire packets."""
    import zmq
    context=zmq.Context();socket=context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE,b'');socket.setsockopt(zmq.RCVHWM,64)
    rearm_stamp=rearm_file.stat().st_mtime_ns if rearm_file and rearm_file.exists() else None
    try:
        start.wait()
        if stop.is_set():return
        # Subscribe only after controller warmup. A connection established
        # during setup would queue seconds of stale poses before the first arm.
        socket.connect(url)
        with (output/'live_packets.jsonl').open('w',encoding='utf8') as log:
            while not stop.is_set():
                if rearm_file and rearm_file.exists():
                    stamp=rearm_file.stat().st_mtime_ns
                    if stamp!=rearm_stamp:
                        packets.put_nowait(dict(kind='explicit_rearm'))
                        log.write(json.dumps(dict(event='explicit_rearm',received_monotonic_ns=time.monotonic_ns()))+'\n')
                        rearm_stamp=stamp
                if not socket.poll(20,zmq.POLLIN):continue
                wire=socket.recv();received=time.monotonic_ns()
                payload=json.loads(wire)
                item=dict(kind='pico_body',received_monotonic_ns=received,payload=payload)
                log.write(json.dumps(item,separators=(',',':'))+'\n')
                packets.put_nowait(item)
    except BaseException:
        (output/'producer_error.txt').write_text(traceback.format_exc())
    finally:socket.close(linger=0);context.term()


def pico_replay_producer(packets,epoch_ns,start,stop,path,output):
    """Replay identical captured packets and arrival times on a declared clock offset."""
    try:
        with np.load(path.parent/'trace.npz',allow_pickle=False) as trace:recorded_epoch=int(trace['epoch_ns'])
        start.wait();offset=recorded_epoch-epoch_ns.value
        with path.open(encoding='utf8') as source,(output/'replayed_packets.jsonl').open('w',encoding='utf8') as log:
            for line in source:
                item=json.loads(line);recorded=int(item['received_monotonic_ns'])
                deadline=epoch_ns.value+recorded-recorded_epoch
                while not stop.is_set():
                    remaining=(deadline-time.monotonic_ns())/1e9
                    if remaining<=0:break
                    stop.wait(min(remaining,.02))
                if stop.is_set():break
                if item.get('event')=='explicit_rearm':delivered=dict(kind='explicit_rearm')
                else:
                    delivered=dict(item,source_clock_offset_ns=offset)
                    if delivered['kind']!='pico_body':raise ValueError('Unexpected captured packet type')
                packets.put_nowait(delivered)
                log.write(json.dumps(dict(recorded=item,replay_delivery_monotonic_ns=time.monotonic_ns(),
                    source_clock_offset_ns=offset),separators=(',',':'))+'\n')
    except BaseException:
        (output/'producer_error.txt').write_text(traceback.format_exc())


def worker(library, name, actor, model, contract, options, initial_packets, packets,
           epoch_ns, start, ready, stop, output, affinity, realtime=False):
    bridge = None
    rows, statuses = [], []
    sensor_rows=[]
    preview_records=[]
    controller=None
    last = -1
    try:
        bridge = Bridge(library, name)
        if realtime:os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(50))
        if affinity and hasattr(os, 'sched_setaffinity'):
            available = sorted(os.sched_getaffinity(0))
            selected={available[min(2,len(available)-1)]}
            if options.get('native_preview_library'):selected=preview_core_affinity(available)
            os.sched_setaffinity(0,selected)
        options=dict(options)
        preview_policy=options.pop('preview_failure_policy','strict')
        feedback=options.pop('feedback','ground-truth')
        sensor_effects=options.pop('sensor_effects',{})
        input_mode=options.pop('input_mode','recorded')
        live_adapter=None;live_initial_arm=input_mode!='recorded';live_rearm_requested=False
        factory_config=options.pop('factory_config',None)
        factory_locomotion=options.pop('factory_locomotion',False)
        locomotion_conditioned=options.pop('locomotion_conditioned',False)
        task_commands=options.pop('task_commands',False)
        native_targets=options.pop('native_targets',False)
        if native_targets:
            from gear_sonic.utils.g1_true23_native_targets import NativeTargetController
            controller=NativeTargetController(actor,factory_config.parents[3],contract,now=-.22,model=model,**options)
        elif task_commands:
            from gear_sonic.utils.g1_true23_task_commands import TaskCommandController
            controller=TaskCommandController(actor,factory_config.parents[3],contract,now=-.22,model=model,**options)
        elif locomotion_conditioned:
            from gear_sonic.utils.g1_true23_locomotion_conditioned import LocomotionConditionedController
            controller=LocomotionConditionedController(actor,factory_config.parents[3],contract,now=-.22,model=model,**options)
        elif factory_locomotion:
            from gear_sonic.utils.g1_true23_factory_locomotion import FactoryLocomotionTeleop
            controller=FactoryLocomotionTeleop(factory_config.parents[3],contract,now=-.22,model=model,onnx_path=actor,**options)
        elif factory_config:
            from gear_sonic.utils.g1_true23_factory_conditioned import NativeFactoryConditionedController
            controller=NativeFactoryConditionedController(actor,factory_config,contract,now=-.22,model=model,**options)
        else:
            controller = CausalController(actor, contract, now=-.22, model=model, **options)
        observation = bridge.observation()
        assert observation is not None and observation[0] == 0
        values = observation[3]
        estimator=None
        measured_q,measured_v=values[:30],values[30:59]
        if feedback=='estimated':
            from gear_sonic.utils.g1_true23_sensor_feedback import Native23SensorFeedback,SensorEffects
            if bridge.observation_width!=462:raise ValueError('Estimated feedback requires the coherent sensor clock')
            estimator=Native23SensorFeedback(model,effects=SensorEffects(**sensor_effects))
            if estimator.effects.delay_controls:
                raise ValueError('Delayed sensors require a stationary measured warmup session; no synthetic padding')
            measured_q,measured_v,sensor_status=estimator.initialize(values[382:439],supported_stationary=True)
            controller.receiver.gate.anchor_pose=measured_q.copy()
            (output/'sensor_feedback_contract.json').write_text(json.dumps(estimator.contract(),indent=2)+'\n')
            sensor_rows.append(np.r_[0.,values[382:439],measured_q,measured_v])
        for sequence, item in enumerate(initial_packets):
            assert controller.receive(*item, (sequence - 11) * .02)
        if estimator is None:
            import_applied_history(controller, values)
        else:
            # A fresh sensor history starts empty. Even initialization must
            # not import a history reconstructed from privileged plant state.
            controller.import_native_history(0)
        # Warm only the ONNX session. No command, history, or blend update.
        if factory_locomotion or locomotion_conditioned or task_commands:
            controller.warmup(measured_q,measured_v,0.,10)
        elif factory_config:
            for _ in range(10):
                before=controller.body_goal.alpha
                controller.command(measured_q,measured_v,0.)
                controller.body_goal.alpha=before
        else:
            features = controller.receiver.features(measured_q, measured_v, 0.)
            if controller.body_goal is not None:
                before = controller.body_goal.alpha
                features = controller.body_goal.features(features, controller.receiver)
                controller.body_goal.alpha = before
            for _ in range(10):
                controller.session.run(None, {'features': features[None]})
        gc.collect()
        gc.disable()
        initial_started=time.monotonic_ns()
        command = controller.command(measured_q, measured_v, 0.)
        (output/'controller_setup.json').write_text(json.dumps(dict(
            affinity_cpus=sorted(os.sched_getaffinity(0)),native_preview_library=str(options.get('native_preview_library'))),indent=2)+'\n')
        target = np.ascontiguousarray(command.targets, np.float64)
        stamp = time.monotonic_ns()
        command_context=dict(control=0,observation_ns=observation[2],started_ns=initial_started,finished_ns=stamp)
        accepted=publish_checked(command,getattr(getattr(controller,'preview_guard',None),'last_diagnostic',None),
            command_context,
            preview_policy,preview_records,
            lambda:publish_native_target(bridge,target,0,initial_started,command_context))
        if not all(accepted):raise RuntimeError('Initial native publication rejected')
        last = 0
        rows.append((0, observation[2], initial_started, command_context['finished_ns'], 1, 1, 11))
        statuses.append(dict(control=0, **command.status))
        ready.set()
        start.wait()
        epoch = epoch_ns.value
        while not stop.is_set():
            # Admission time is actual worker receipt, not producer send time.
            while True:
                try:
                    delivered = packets.get_nowait()
                except queue.Empty:
                    break
                now = (time.monotonic_ns() - epoch) / 1e9
                if isinstance(delivered,dict):
                    if delivered['kind']=='explicit_rearm':
                        live_rearm_requested=True
                        continue
                    if delivered['kind']!='pico_body':raise ValueError('Unknown live input event')
                    from gear_sonic.utils.g1_true23_pico_body_adapter import Native23PicoBodyAdapter
                    if live_initial_arm or live_rearm_requested:
                        # Initial --input live is one explicit session start.
                        # Faults require a separate operator rearm event.
                        controller.rearm(now,measured_q)
                        live_adapter=Native23PicoBodyAdapter(contract,epoch=controller.receiver.gate.epoch)
                        live_initial_arm=live_rearm_requested=False
                    try:
                        mapped=live_adapter.receive(controller,delivered['payload'],
                            received_monotonic_ns=time.monotonic_ns()+int(delivered.get('source_clock_offset_ns',0)),now=now)
                        statuses.append(dict(event='pico_body_received',**mapped.source))
                    except ValueError as exc:
                        statuses.append(dict(event='pico_body_rejected',error=str(exc),explicit_rearm_required=True))
                else:
                    sent,item=delivered
                    controller.receive(*item, now)
            observation = bridge.observation()
            if observation is None or observation[0] <= last:
                if hasattr(bridge.lib,'clock_wait_observation'):
                    bridge.lib.clock_wait_observation(bridge.address,last,5_000_000)
                else:time.sleep(.0001)
                continue
            control, fault, published, values = observation
            started = time.monotonic_ns()
            now = (started - epoch) / 1e9
            if fault:
                controller.receiver.gate.latch('native_clock_deadline_fault', now)
            if control != last + 1:
                controller.receiver.gate.latch('missed_robot_observation', now)
                bridge.lib.clock_latch_fault(bridge.address)
            last = control
            if estimator is None:
                import_applied_history(controller, values, control)
                measured_q,measured_v=values[:30],values[30:59]
            else:
                # Applied target belongs to the plant; sensor history belongs
                # to the estimator. Never import GT gyro, gravity or joint history.
                actual_target=values[439:462].copy()
                controller.receiver.history.commit(measured_q,measured_v,actual_target)
                measured_q,measured_v,sensor_status=estimator.update(values[382:439],now=now)
                controller.import_native_history(control)
                controller.previous_native=actual_target
                sensor_rows.append(np.r_[float(control),values[382:439],measured_q,measured_v])
            command = controller.command(measured_q, measured_v, now)
            if estimator is not None:command.status.update(sensor_status)
            target = np.ascontiguousarray(command.targets, np.float64)
            finished = time.monotonic_ns()
            command_context=dict(control=control,observation_ns=published,started_ns=started,finished_ns=finished)
            accepted=publish_checked(command,getattr(getattr(controller,'preview_guard',None),'last_diagnostic',None),
                command_context,
                preview_policy,preview_records,
                lambda:publish_native_target(bridge,target,control,started,command_context))
            rows.append((control, published, started, command_context['finished_ns'], *accepted, controller.receiver.gate.last_sequence))
            statuses.append(dict(control=control, **command.status))
    except SimulatorPreviewRejected as exc:
        exc.record['stop_requested_ns']=stop_simulator(bridge,stop)
        (output/'preview_rejection.json').write_text(json.dumps(json_safe(exc.record),indent=2,allow_nan=False)+'\n')
        ready.set()
    except BaseException:
        stop_simulator(bridge,stop)
        (output / 'worker_error.txt').write_text(traceback.format_exc())
        ready.set()
    finally:
        if controller is not None:
            (output/'packet_report.json').write_text(json.dumps(controller.receiver.gate.epoch_report(),indent=2)+'\n')
            if controller.receiver.stop is not None:
                np.save(output/'fault_standing_goal.npy',controller.receiver.stop.last)
        # Local worker-to-parent evidence only; never deserialize external input.
        with (output/'preview_records.local.pkl').open('wb') as stream:
            pickle.dump(preview_records,stream)
        if sensor_rows:np.save(output/'sensor_feedback.npy',np.asarray(sensor_rows))
        np.save(output / 'controller_calls.npy', np.asarray(rows, np.int64).reshape(-1, 7))
        (output / 'controller_status.json').write_text(json.dumps(statuses, indent=2) + '\n')
        if bridge is not None:
            bridge.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path, default=NEW / 'causal_dynamics_v1/bank')
    parser.add_argument('--clip', choices=['walk002', 'walk003', 'pico', 'walk008'], default='walk002')
    parser.add_argument('--controls', type=int)
    parser.add_argument('--hold', type=int, default=30)
    parser.add_argument('--standing', action='store_true')
    parser.add_argument('--initial-velocity', type=float, default=0.)
    parser.add_argument('--fault-control', type=int)
    parser.add_argument('--affinity', action='store_true')
    parser.add_argument('--native-conditioned',action='store_true')
    parser.add_argument('--factory-locomotion',action='store_true')
    parser.add_argument('--locomotion-conditioned',action='store_true')
    parser.add_argument('--task-commands',action='store_true')
    parser.add_argument('--native-targets',action='store_true')
    parser.add_argument('--native-preview-guard',action='store_true')
    parser.add_argument('--preview-failure-policy',choices=['strict','diagnostic-only'],default='strict')
    parser.add_argument('--native-standing-capture',action='store_true')
    parser.add_argument('--native-preview-delay-substeps',type=int,choices=range(7),default=0)
    parser.add_argument('--native-preview-library',type=Path)
    parser.add_argument('--feedback',choices=['ground-truth','estimated'],default='ground-truth')
    parser.add_argument('--sensor-effects',type=Path,help='Declared sensor noise/bias configuration JSON')
    parser.add_argument('--input',choices=['recorded','live'],default='recorded')
    parser.add_argument('--packet-replay',type=Path,help='Recorded live_packets.jsonl with adjacent original trace.npz')
    parser.add_argument('--live-url',default='tcp://127.0.0.1:5557')
    parser.add_argument('--rearm-file',type=Path,help='Explicit live-input rearm when this operator-owned file changes')
    parser.add_argument('--target-filter-alpha',type=float,default=1.)
    parser.add_argument('--fault-standing-capture',action='store_true')
    parser.add_argument('--plant-spin-us',type=int,default=0)
    parser.add_argument('--lookahead-library',type=Path)
    parser.add_argument('--lookahead-steps',type=int,default=30)
    parser.add_argument('--factory-config',type=Path)
    parser.add_argument('--realtime-priority',action='store_true',help='Use bounded-process Linux FIFO priorities70/50/40 for plant/controller/producer')
    parser.add_argument('--library',type=Path,default=NEW/'native_clock_v1/libtrue23clock.so')
    args = parser.parse_args()
    pico_input=args.input=='live' or args.packet_replay is not None
    if args.input=='live' and args.packet_replay is not None:raise ValueError('Choose live input or captured packet replay')
    if args.preview_failure_policy=='diagnostic-only' and not args.native_preview_guard:
        raise ValueError('Diagnostic-only policy requires native preview guard')
    if args.native_preview_guard and not args.native_targets:raise ValueError('Native preview requires --native-targets')
    if args.native_standing_capture and not args.native_targets:raise ValueError('Native standing capture requires --native-targets')
    if (args.native_preview_delay_substeps or args.native_preview_library) and not args.native_preview_guard:
        raise ValueError('Native preview settings require --native-preview-guard')
    if args.native_targets and not args.task_commands:raise ValueError('Native target actuation requires --task-commands and a native-target actor')
    if sum((args.factory_locomotion,args.native_conditioned,args.locomotion_conditioned,args.task_commands))>1:raise ValueError('select only one factory controller')
    if args.lookahead_library and not args.locomotion_conditioned:raise ValueError('short lookahead requires the received locomotion-conditioned controller')
    if args.target_filter_alpha!=1 and not args.locomotion_conditioned:raise ValueError('Target filtering requires --locomotion-conditioned')
    if args.fault_standing_capture and not (args.locomotion_conditioned or args.task_commands):raise ValueError('Fault standing capture requires a received conditioned controller')
    if not 0<=args.plant_spin_us<=500:raise ValueError('Plant spin must be0..500 microseconds')
    locomotion_enabled=args.factory_locomotion or args.locomotion_conditioned or args.task_commands
    factory_enabled=locomotion_enabled or args.native_conditioned
    if args.factory_config is None:
        cfg='human_loco/fsm_human_loco_config.yaml' if locomotion_enabled else 'mimic_test/fsm_mimic_test.yaml'
        args.factory_config=NEW/'onboard_factory_firmware_v1/decoded_configs/policies'/cfg
    if sys.platform != 'linux' or mujoco.__version__ != '3.2.3':
        raise RuntimeError('Use the Linux/WSL native MuJoCo3.2.3 interpreter')
    args.output.mkdir(parents=True, exist_ok=False)
    pin_run(args)
    model, contract, motion, original, timeline = load_case(args.clip)
    bank = archive(args.bank / (args.clip + '.npz'))
    meta = json.loads((args.bank / 'bank.json').read_text())
    total = timeline['total_requested_controls']
    controls = args.controls or (3000 if pico_input else (1500 if args.standing else total + args.hold * 50))
    if controls < 1 or controls > 2_147_483_647 // 10:
        raise ValueError('Controls must be positive and fit the native signed 32-bit physics step count')
    if args.fault_control is not None and args.fault_control < 1:
        raise ValueError('fault-control must follow initialized control0')
    data = mujoco.MjData(model)
    initial = bank['states'][10].copy()
    initial[30] += args.initial_velocity
    data.qpos[:] = initial[:30]
    data.qvel[:] = initial[30:]
    mujoco.mj_forward(model, data)
    library = args.library
    bridge = Bridge(library)
    bridge.lib.clock_set_spin_ns(args.plant_spin_us*1000)
    if factory_enabled:
        bridge.lib.clock_set_limit_brake.argtypes=[ct.c_int]
        bridge.lib.clock_set_limit_brake(int(not args.native_targets))
    values = np.zeros(bridge.observation_width,np.float64)
    values[:59]=initial
    if bridge.observation_width==462:
        values[382:439]=initial_sensor_sample(model,data)
        values[439:462]=initial[7:30]
    bridge.publish_observation(bridge.address, 0, 0, time.monotonic_ns(), ptr(values))
    context = mp.get_context('fork')
    stop, start, ready = context.Event(), context.Event(), context.Event()
    epoch_ns = context.Value(ct.c_int64, 0)
    packets = context.Queue(maxsize=64)
    initial_packets = [packet_at(motion, original, sequence,args.standing or pico_input,
                                final=pico_input and sequence==11) for sequence in range(12)]
    options = dict(standing_qpos=timeline['configured_standing_qpos'], tasks=meta['tasks'])
    options['preview_failure_policy']=args.preview_failure_policy
    options['feedback']=args.feedback
    options['input_mode']='pico-replay' if args.packet_replay is not None else args.input
    options['sensor_effects']={} if args.sensor_effects is None else json.loads(args.sensor_effects.read_text())
    if args.feedback=='estimated' and not args.native_targets:raise ValueError('Sensor integration targets the current native23 controller')
    if factory_enabled:
        options['factory_config']=args.factory_config;options['factory_locomotion']=args.factory_locomotion
        options['locomotion_conditioned']=args.locomotion_conditioned
        options['task_commands']=args.task_commands
        options['native_targets']=args.native_targets
        if args.native_targets:options['native_preview_guard']=args.native_preview_guard
        if args.native_targets:options['native_standing_capture']=args.native_standing_capture
        if args.native_targets:
            options['native_preview_delay_substeps']=args.native_preview_delay_substeps
            options['native_preview_library']=args.native_preview_library
        if args.locomotion_conditioned:options['target_filter_alpha']=args.target_filter_alpha
        if args.locomotion_conditioned or args.task_commands:options['fault_standing_capture']=args.fault_standing_capture
        if args.lookahead_library:options.update(lookahead_library=args.lookahead_library,lookahead_steps=args.lookahead_steps)
    controller = context.Process(target=worker, args=(library, bridge.shm.name, args.actor, model, contract,
        options, initial_packets, packets, epoch_ns, start, ready, stop, args.output, args.affinity,args.realtime_priority))
    sender = (context.Process(target=live_producer,args=(packets,start,stop,args.live_url,args.output,args.rearm_file))
        if args.input=='live' else context.Process(target=producer, args=(packets, epoch_ns, start, stop, motion, original,
        controls, args.standing, args.fault_control, args.output,args.realtime_priority,library)))
    if args.packet_replay is not None:
        sender=context.Process(target=pico_replay_producer,args=(packets,epoch_ns,start,stop,args.packet_replay,args.output))
    processes = [controller, sender]
    for process in processes:
        process.start()
    plant_contract=dict(contract)
    if factory_enabled and not args.native_targets:
        import yaml
        factory=yaml.safe_load(args.factory_config.read_text(encoding='utf-8'))
        plant_contract['kp']=np.asarray(factory['joint_kp'])
        plant_contract['kd']=np.asarray(factory['joint_kd'])
        if locomotion_enabled:
            from gear_sonic.utils.g1_true23_factory_locomotion import IDS
            plant_contract['kp']=plant_contract['kp'][IDS];plant_contract['kd']=plant_contract['kd'][IDS]
        # The ABI uses effort/kp only to normalize stored command history.
        # Keep the original receiver convention despite different PD gains.
        plant_contract['training_effort']=np.asarray(contract['training_effort'])*plant_contract['kp']/contract['kp']
    count=0
    states=np.r_[initial,0.][None]
    targets=np.empty((0,23));torques=np.empty((0,23));timing=np.empty((0,7),np.int64)
    summary=np.zeros(10);summary[4]=-1
    setup_failure=None
    native_return_ns=None
    try:
        if not ready.wait(60) or (args.output / 'worker_error.txt').exists():
            raise RuntimeError('Controller setup failed; see worker_error.txt')
        rejection_path=args.output/'preview_rejection.json'
        if rejection_path.exists():raise SimulatorPreviewRejected(json.loads(rejection_path.read_text()))
        result = bridge.result()
        if result is None or result[0] != 0:
            raise RuntimeError('No warmed initial command')
        target = result[1]
        states = np.empty((controls * 10 + 1, 60))
        targets, torques = np.empty((controls * 10, 23)), np.empty((controls * 10, 23))
        timing, summary = np.empty((controls * 10, 7), np.int64), np.zeros(10)
        arrays = [np.ascontiguousarray(plant_contract[k], np.float64) for k in
                  ('kp', 'kd', 'native_effort', 'native_velocity', 'default_q', 'training_effort')]
        if args.affinity:
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        if args.realtime_priority:os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(70))
        epoch_ns.value = time.monotonic_ns() + 100_000_000
        start.set()
        count = bridge.lib.clock_run(bridge.address, model._address, data._address, controls, epoch_ns.value,
            *[ptr(x) for x in arrays], ptr(target), ptr(states), ptr(targets), ptr(torques), timing.ctypes.data_as(INT64), ptr(summary))
        native_return_ns=time.monotonic_ns()
    except SimulatorPreviewRejected:
        # The worker already stopped the bridge. Keep a normal zero-step report.
        pass
    except BaseException:
        setup_failure=traceback.format_exc()
        (args.output/'runner_error.txt').write_text(setup_failure)
    finally:
        if args.realtime_priority:os.sched_setscheduler(0,os.SCHED_OTHER,os.sched_param(0))
        stop.set()
        start.set()
        bridge.lib.clock_stop(bridge.address)
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(5)
        bridge.close(owner=True)
        packets.close()
    rejection_path=args.output/'preview_rejection.json'
    rejection=json.loads(rejection_path.read_text()) if rejection_path.exists() else None
    if count==0:
        states=np.r_[initial,0.][None]
        summary=np.zeros(10);summary[4]=-1
    states, targets, torques, timing = states[:count + 1], targets[:count], torques[:count], timing[:count]
    if rejection is not None:
        rejection['native_return_ns']=native_return_ns
        rejection['last_physics_finish_ns']=int(timing[-1,3]) if count else None
        rejection['physics_steps_finishing_after_stop_request']=int(np.count_nonzero(timing[:,3]>rejection['stop_requested_ns']))
        rejection_path.write_text(json.dumps(rejection,indent=2)+'\n')
    np.savez_compressed(args.output / 'trace.npz', states=states, targets=targets, torques=torques,
                        timing=timing, epoch_ns=np.asarray(epoch_ns.value))
    records_path=args.output/'preview_records.local.pkl'
    if records_path.exists():
        with records_path.open('rb') as stream:preview_records=pickle.load(stream)
        write_diagnostics(args.output/'preview_diagnostics.jsonl',preview_records,timing,epoch_ns.value)
        records_path.unlink()
    complete = count == controls * 10 and not bool(summary[3]) and rejection is None and setup_failure is None
    steps = count // 10
    control_ids = np.arange(steps)
    frames = np.minimum(control_ids + 11, len(motion['joint_pos']) - 1)
    source = dict(passed=False, source_controls=0,live_packet_evaluation_required=pico_input) if args.standing or pico_input else metrics(model, states[(control_ids + 1) * 10, :30],
        frames, control_ids, motion, original, timeline, meta['tasks'])
    goal = np.r_[motion['body_pos_w'][11, 0], motion['body_quat_w'][11, 0], motion['joint_pos'][11]] if args.standing or pico_input else original['source_qpos29'][-1]
    final_quiet = quiet(states[:, :30], states[:, 30:59], goal)
    continuous_quiet30 = quiet(states[:, :30], states[:, 30:59], goal,seconds=30)
    main_quiet = quiet(states[:total * 10 + 1, :30], states[:total * 10 + 1, 30:59], goal) if steps >= total and not args.standing else None
    calls_path=args.output/'controller_calls.npy'
    calls = np.load(calls_path) if calls_path.exists() else np.empty((0,7),np.int64)
    packet_path = args.output / 'packet_report.json'
    packet_report = json.loads(packet_path.read_text()) if packet_path.exists() else None
    trailing_quiet, trailing_quiet30=final_quiet,continuous_quiet30
    hold_reached=bool(complete and not pico_input and
        ((args.standing and steps>=1500) or (not args.standing and steps>=total+1500
         and source.get('source_controls',0)==source.get('requested_source_controls',-1)
         and packet_report is not None and packet_report.get('fault') is None
         and packet_report.get('full_source_consumed') is True)))
    final_quiet,continuous_quiet30=terminal_standing(trailing_quiet,trailing_quiet30,hold_reached)
    finish_late = timing[:, 3] - timing[:, 0]
    activation_lags, mixed_controls = [], 0
    for control in range(steps):
        block = timing[control * 10:(control + 1) * 10]
        active = np.flatnonzero(block[:, 4] == control)
        if control and len(active):
            activation_lags.append((block[active[0], 0] - epoch_ns.value - control * 20_000_000) * 1e-6)
        mixed_controls += len(np.unique(block[:, 4])) > 1
    timing_pass = (summary[1] == 0 and summary[2] == 0 and summary[9] == 0 and not np.any(finish_late > 2_000_000))
    worker_ok = not any((args.output / name).exists() for name in ('worker_error.txt', 'producer_error.txt','runner_error.txt'))
    packet_ok = packet_report is not None and packet_report['fault'] is None
    fault_goal = args.output / 'fault_standing_goal.npy'
    fault_quiet = quiet(states[:, :30], states[:, 30:59], np.load(fault_goal)) if fault_goal.exists() else None
    fault_continuous_quiet30 = quiet(states[:, :30], states[:, 30:59], np.load(fault_goal),seconds=30) if fault_goal.exists() else None
    behavior_pass = final_quiet['passed'] if args.standing else source['passed'] and main_quiet is not None and main_quiet['passed'] and final_quiet['passed'] and steps >= total + 1500
    behavior_pass=behavior_pass and continuous_quiet30['passed']
    stop_reason=('preview_rejected' if rejection else 'worker_error' if not worker_ok else
                 'native_physical_limit' if summary[3] else 'complete' if complete else 'native_stop')
    setup_path=args.output/'controller_setup.json'
    setup=json.loads(setup_path.read_text()) if setup_path.exists() else {}
    report = dict(runner_contract_version=RUNNER_CONTRACT_VERSION,
        preview_failure_policy=args.preview_failure_policy,preview_rejection=rejection,
        stop_reason=stop_reason,actual_stop_simulation_seconds=float(states[-1,-1]),
        terminal_standing_reached=hold_reached,trailing_quiet_diagnostic=trailing_quiet,
        trailing_quiet30_diagnostic=trailing_quiet30,
        actor=str(args.actor), clip=args.clip, standing_diagnostic=args.standing,
        input_mode=args.input,feedback_mode=args.feedback,
        recorded_pico_packet_source=str(args.packet_replay) if args.packet_replay else None,
        live_packet_log=str(args.output/'live_packets.jsonl') if args.input=='live' else None,
        native_factory_conditioned=args.native_conditioned,factory_locomotion=args.factory_locomotion,
        locomotion_conditioned=args.locomotion_conditioned,task_commands=args.task_commands,
        native_target_actuation=args.native_targets,native_limit_brake=factory_enabled and not args.native_targets,
        native_preview_guard=args.native_preview_guard,
        native_standing_capture=args.native_standing_capture,
        native_preview_delay_substeps=args.native_preview_delay_substeps,
        native_preview_library=str(args.native_preview_library) if args.native_preview_library else None,
        controller_affinity_cpus=setup.get('affinity_cpus',[]),
        native_snapshot_protocol=(bridge.lib.clock_snapshot_protocol_version() if hasattr(bridge.lib,'clock_snapshot_protocol_version') else 0),
        target_margin_rad=0. if args.native_targets else (.06 if factory_enabled else None),
        leg_target_filter_alpha=1. if args.native_targets else (.9 if args.task_commands else args.target_filter_alpha),
        factory_prior_filter_alpha=.9 if args.native_targets else None,
        initial_world_velocity_x_delta_mps=args.initial_velocity,
        fault_standing_capture_enabled=args.fault_standing_capture,
        plant_spin_us=args.plant_spin_us,
        native_lookahead_library=str(args.lookahead_library) if args.lookahead_library else None,
        native_lookahead_steps=args.lookahead_steps if args.lookahead_library else 0,
        linux_fifo_priorities=[70,50,40] if args.realtime_priority else None,
        producer_gc_disabled_during_replay=True,
        producer_wait='CLOCK_MONOTONIC absolute native sleep' if hasattr(bridge.lib,'clock_wait_until') else 'Python Event timed wait',
        controller_boundary_wakeup='native shared futex' if hasattr(bridge.lib,'clock_wait_observation') else 'polling',
        native_library=str(library),actual_pd_kp=plant_contract['kp'].tolist(),actual_pd_kd=plant_contract['kd'].tolist(),
        requested_controls=controls, completed_controls=steps, physical_steps=count, physical_complete=complete,
        physical_failure=bool(summary[3]), controller_deadline_misses=int(summary[1]),
        plant_wakes_over2ms_late=int(summary[2]), plant_finishes_over2ms_late=int(np.count_nonzero(finish_late > 2_000_000)),
        maximum_plant_finish_lateness_ms=float(finish_late.max() * 1e-6) if count else None,
        missed_observation_publications=int(summary[9]), first_latched_plant_fault_control=int(summary[4]),
        maximum_speed_ratio=float(summary[5]), maximum_range_excess=float(summary[6]), maximum_effort_ratio=float(summary[7]),
        controller_ms_p50_p95_max=np.percentile((calls[1:, 3] - calls[1:, 2]) * 1e-6, [50, 95, 100]).tolist() if len(calls) > 1 else [],
        first_target_application_ms_p50_p95_max=np.percentile(activation_lags, [50, 95, 100]).tolist() if activation_lags else [],
        controls_with_previous_then_current_command=int(mixed_controls),
        activation_schedule='new command applies at next 2ms step after publication; previous target stays applied during computation',
        tracking_passed=bool(source['passed']), timing_passed=bool(timing_pass and count>0),
        standing_passed=bool(hold_reached and continuous_quiet30['passed']),
        source=source, main_quiet=main_quiet, final_quiet=final_quiet,continuous_quiet30=continuous_quiet30,packet_report=packet_report,
        passed=bool(complete and behavior_pass and timing_pass and worker_ok and packet_ok and args.preview_failure_policy=='strict'),
        one_learned_controller=True, identical_targets_in_both_legacy_channels=True,
        applied_history_owner=('native applied targets plus sensor-only history' if args.feedback=='estimated'
            else 'native plant; imported once per observed control boundary'),
        independent_physics_500hz=True, independent_packet_producer_50hz=True, controller_hz=50,
        warmed_onnx_calls=10, physics_steps_during_warmup=0, preclock_received_samples=12,
        simulation_time_frozen_for_inference=False, future_reference_frames=0, prepared_motion_specific=False,
        injected_input_loss_control=args.fault_control, explicit_rearm_api_preserved=True, automatic_rearm=False,
        fault_quiet=fault_quiet,
        fault_continuous_quiet30=fault_continuous_quiet30,
        fault_scenario_passed=bool(args.fault_control is not None and complete and timing_pass and worker_ok
            and packet_report is not None and packet_report['fault'] is not None and fault_quiet and fault_quiet['passed']
            and fault_continuous_quiet30 and fault_continuous_quiet30['passed']),
        native_clock_long_pico_recurrence_unresolved=False,
        native_clock_expected_time='Independent repeated additions of .002 from zero; tolerance 1e-10',
        general_live_teleop_qualified=False, hardware_authorized=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
