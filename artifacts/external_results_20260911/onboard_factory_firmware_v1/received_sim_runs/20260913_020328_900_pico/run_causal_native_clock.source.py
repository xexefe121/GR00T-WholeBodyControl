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

DOUBLE = ct.POINTER(ct.c_double)
INT64 = ct.POINTER(ct.c_int64)


def ptr(array):
    return array.ctypes.data_as(DOUBLE)


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
        values = np.empty(382)
        if self.lib.clock_read_observation(self.address, ct.byref(ident), ct.byref(fault), ct.byref(stamp), ptr(values)):
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


def producer(packets, epoch_ns, start, stop, motion, original, controls, standing, fault_control, output, realtime=False):
    rows = []
    try:
        if realtime:os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(40))
        # Packet views/dicts are reference-counted and do not form cycles.
        # Collect the inherited interpreter heap before the clock starts;
        # never pause recorded packet delivery for a full-heap GC traversal.
        gc.collect();gc.disable()
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
                stop.wait(min(remaining, .02))
            if stop.is_set():
                break
            stamp = time.monotonic_ns()
            item = packet_at(motion, original, sequence, standing, sequence == last_sequence)
            try:
                packets.put_nowait((stamp, item))
                accepted = 1
            except queue.Full:
                accepted = 0
            rows.append((sequence, deadline, stamp, accepted))
    except BaseException:
        (output / 'producer_error.txt').write_text(traceback.format_exc())
    finally:
        np.save(output / 'producer_calls.npy', np.asarray(rows, np.int64).reshape(-1, 4))


def worker(library, name, actor, model, contract, options, initial_packets, packets,
           epoch_ns, start, ready, stop, output, affinity, realtime=False):
    bridge = None
    rows, statuses = [], []
    last = -1
    try:
        bridge = Bridge(library, name)
        if realtime:os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(50))
        if affinity and hasattr(os, 'sched_setaffinity'):
            available = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {available[min(2, len(available) - 1)]})
        options=dict(options)
        factory_config=options.pop('factory_config',None)
        factory_locomotion=options.pop('factory_locomotion',False)
        locomotion_conditioned=options.pop('locomotion_conditioned',False)
        if locomotion_conditioned:
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
        for sequence, item in enumerate(initial_packets):
            assert controller.receive(*item, (sequence - 11) * .02)
        observation = bridge.observation()
        assert observation is not None and observation[0] == 0
        values = observation[3]
        import_applied_history(controller, values)
        # Warm only the ONNX session. No command, history, or blend update.
        if factory_locomotion or locomotion_conditioned:
            controller.warmup(values[:30],values[30:59],0.,10)
        elif factory_config:
            for _ in range(10):
                before=controller.body_goal.alpha
                controller.command(values[:30],values[30:59],0.)
                controller.body_goal.alpha=before
        else:
            features = controller.receiver.features(values[:30], values[30:59], 0.)
            if controller.body_goal is not None:
                before = controller.body_goal.alpha
                features = controller.body_goal.features(features, controller.receiver)
                controller.body_goal.alpha = before
            for _ in range(10):
                controller.session.run(None, {'features': features[None]})
        gc.collect()
        gc.disable()
        command = controller.command(values[:30], values[30:59], 0.)
        target = np.ascontiguousarray(command.targets, np.float64)
        stamp = time.monotonic_ns()
        for channel in (0, 1):
            assert bridge.lib.clock_publish_result(bridge.address, channel, 0, stamp, stamp, ptr(target))
        last = 0
        rows.append((0, observation[2], stamp, stamp, 1, 1, 11))
        statuses.append(dict(control=0, **command.status))
        ready.set()
        start.wait()
        epoch = epoch_ns.value
        while not stop.is_set():
            # Admission time is actual worker receipt, not producer send time.
            while True:
                try:
                    sent, item = packets.get_nowait()
                except queue.Empty:
                    break
                now = (time.monotonic_ns() - epoch) / 1e9
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
            import_applied_history(controller, values, control)
            command = controller.command(values[:30], values[30:59], now)
            target = np.ascontiguousarray(command.targets, np.float64)
            finished = time.monotonic_ns()
            accepted = [bridge.lib.clock_publish_result(bridge.address, channel, control, started, finished, ptr(target))
                        for channel in (0, 1)]
            rows.append((control, published, started, finished, *accepted, controller.receiver.gate.last_sequence))
            statuses.append(dict(control=control, **command.status))
        (output / 'packet_report.json').write_text(json.dumps(controller.receiver.gate.epoch_report(), indent=2) + '\n')
        if controller.receiver.stop is not None:
            np.save(output / 'fault_standing_goal.npy', controller.receiver.stop.last)
    except BaseException:
        (output / 'worker_error.txt').write_text(traceback.format_exc())
        if bridge is not None:
            bridge.lib.clock_latch_fault(bridge.address)
        ready.set()
    finally:
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
    parser.add_argument('--factory-config',type=Path)
    parser.add_argument('--realtime-priority',action='store_true',help='Use bounded-process Linux FIFO priorities70/50/40 for plant/controller/producer')
    parser.add_argument('--library',type=Path,default=NEW/'native_clock_v1/libtrue23clock.so')
    args = parser.parse_args()
    if sum((args.factory_locomotion,args.native_conditioned,args.locomotion_conditioned))>1:raise ValueError('select only one factory controller')
    locomotion_enabled=args.factory_locomotion or args.locomotion_conditioned
    factory_enabled=locomotion_enabled or args.native_conditioned
    if args.factory_config is None:
        cfg='human_loco/fsm_human_loco_config.yaml' if locomotion_enabled else 'mimic_test/fsm_mimic_test.yaml'
        args.factory_config=NEW/'onboard_factory_firmware_v1/decoded_configs/policies'/cfg
    if sys.platform != 'linux' or mujoco.__version__ != '3.2.3':
        raise RuntimeError('Use the Linux/WSL native MuJoCo3.2.3 interpreter')
    args.output.mkdir(parents=True, exist_ok=False)
    model, contract, motion, original, timeline = load_case(args.clip)
    bank = archive(args.bank / (args.clip + '.npz'))
    meta = json.loads((args.bank / 'bank.json').read_text())
    total = timeline['total_requested_controls']
    controls = args.controls or (1500 if args.standing else total + args.hold * 50)
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
    bridge.lib.clock_set_spin_ns(0)
    if factory_enabled:
        bridge.lib.clock_set_limit_brake.argtypes=[ct.c_int]
        bridge.lib.clock_set_limit_brake(1)
    values = np.ascontiguousarray(np.r_[initial, np.zeros(323)], np.float64)
    bridge.lib.clock_publish_observation(bridge.address, 0, 0, time.monotonic_ns(), ptr(values))
    context = mp.get_context('fork')
    stop, start, ready = context.Event(), context.Event(), context.Event()
    epoch_ns = context.Value(ct.c_int64, 0)
    packets = context.Queue(maxsize=64)
    initial_packets = [packet_at(motion, original, sequence, args.standing) for sequence in range(12)]
    options = dict(standing_qpos=timeline['configured_standing_qpos'], tasks=meta['tasks'])
    if factory_enabled:
        options['factory_config']=args.factory_config;options['factory_locomotion']=args.factory_locomotion
        options['locomotion_conditioned']=args.locomotion_conditioned
    controller = context.Process(target=worker, args=(library, bridge.shm.name, args.actor, model, contract,
        options, initial_packets, packets, epoch_ns, start, ready, stop, args.output, args.affinity,args.realtime_priority))
    sender = context.Process(target=producer, args=(packets, epoch_ns, start, stop, motion, original,
        controls, args.standing, args.fault_control, args.output,args.realtime_priority))
    processes = [controller, sender]
    for process in processes:
        process.start()
    try:
        if not ready.wait(60) or (args.output / 'worker_error.txt').exists():
            raise RuntimeError('Controller setup failed; see worker_error.txt')
        result = bridge.result()
        if result is None or result[0] != 0:
            raise RuntimeError('No warmed initial command')
        target = result[1]
        states = np.empty((controls * 10 + 1, 60))
        targets, torques = np.empty((controls * 10, 23)), np.empty((controls * 10, 23))
        timing, summary = np.empty((controls * 10, 7), np.int64), np.empty(10)
        plant_contract=dict(contract)
        if factory_enabled:
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
        arrays = [np.ascontiguousarray(plant_contract[k], np.float64) for k in
                  ('kp', 'kd', 'native_effort', 'native_velocity', 'default_q', 'training_effort')]
        if args.affinity:
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        if args.realtime_priority:os.sched_setscheduler(0,os.SCHED_FIFO,os.sched_param(70))
        epoch_ns.value = time.monotonic_ns() + 100_000_000
        start.set()
        count = bridge.lib.clock_run(bridge.address, model._address, data._address, controls, epoch_ns.value,
            *[ptr(x) for x in arrays], ptr(target), ptr(states), ptr(targets), ptr(torques), timing.ctypes.data_as(INT64), ptr(summary))
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
    states, targets, torques, timing = states[:count + 1], targets[:count], torques[:count], timing[:count]
    np.savez_compressed(args.output / 'trace.npz', states=states, targets=targets, torques=torques,
                        timing=timing, epoch_ns=np.asarray(epoch_ns.value))
    complete = count == controls * 10 and not bool(summary[3])
    steps = count // 10
    control_ids = np.arange(steps)
    frames = np.minimum(control_ids + 11, len(motion['joint_pos']) - 1)
    source = dict(passed=False, source_controls=0) if args.standing else metrics(model, states[(control_ids + 1) * 10, :30],
        frames, control_ids, motion, original, timeline, meta['tasks'])
    goal = np.r_[motion['body_pos_w'][11, 0], motion['body_quat_w'][11, 0], motion['joint_pos'][11]] if args.standing else original['source_qpos29'][-1]
    final_quiet = quiet(states[:, :30], states[:, 30:59], goal)
    main_quiet = quiet(states[:total * 10 + 1, :30], states[:total * 10 + 1, 30:59], goal) if steps >= total and not args.standing else None
    calls = np.load(args.output / 'controller_calls.npy')
    packet_path = args.output / 'packet_report.json'
    packet_report = json.loads(packet_path.read_text()) if packet_path.exists() else None
    finish_late = timing[:, 3] - timing[:, 0]
    activation_lags, mixed_controls = [], 0
    for control in range(steps):
        block = timing[control * 10:(control + 1) * 10]
        active = np.flatnonzero(block[:, 4] == control)
        if control and len(active):
            activation_lags.append((block[active[0], 0] - epoch_ns.value - control * 20_000_000) * 1e-6)
        mixed_controls += len(np.unique(block[:, 4])) > 1
    timing_pass = (summary[1] == 0 and summary[2] == 0 and summary[9] == 0 and not np.any(finish_late > 2_000_000))
    worker_ok = not any((args.output / name).exists() for name in ('worker_error.txt', 'producer_error.txt'))
    packet_ok = packet_report is not None and packet_report['fault'] is None
    fault_goal = args.output / 'fault_standing_goal.npy'
    fault_quiet = quiet(states[:, :30], states[:, 30:59], np.load(fault_goal)) if fault_goal.exists() else None
    behavior_pass = final_quiet['passed'] if args.standing else source['passed'] and main_quiet is not None and main_quiet['passed'] and final_quiet['passed'] and steps >= total + 1500
    report = dict(actor=str(args.actor), clip=args.clip, standing_diagnostic=args.standing,
        native_factory_conditioned=args.native_conditioned,factory_locomotion=args.factory_locomotion,
        locomotion_conditioned=args.locomotion_conditioned,native_limit_brake=factory_enabled,
        linux_fifo_priorities=[70,50,40] if args.realtime_priority else None,
        producer_gc_disabled_during_replay=True,
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
        source=source, main_quiet=main_quiet, final_quiet=final_quiet, packet_report=packet_report,
        passed=bool(complete and behavior_pass and timing_pass and worker_ok and packet_ok),
        one_learned_controller=True, identical_targets_in_both_legacy_channels=True,
        applied_history_owner='unchanged native plant; imported once per observed control boundary',
        independent_physics_500hz=True, independent_packet_producer_50hz=True, controller_hz=50,
        warmed_onnx_calls=10, physics_steps_during_warmup=0, preclock_received_samples=12,
        simulation_time_frozen_for_inference=False, future_reference_frames=0, prepared_motion_specific=False,
        injected_input_loss_control=args.fault_control, explicit_rearm_api_preserved=True, automatic_rearm=False,
        fault_quiet=fault_quiet,
        fault_scenario_passed=bool(args.fault_control is not None and complete and timing_pass and worker_ok
            and packet_report is not None and packet_report['fault'] is not None and fault_quiet and fault_quiet['passed']),
        native_clock_long_pico_recurrence_unresolved=False,
        native_clock_expected_time='Independent repeated additions of .002 from zero; tolerance 1e-10',
        general_live_teleop_qualified=False, hardware_authorized=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.copy2(__file__, args.output / 'run_causal_native_clock.source.py')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
