"""ONE offline walk002 saved-MPC-prefix / actual BFM terminal hybrid.

Fresh canonical native physics. Original applied MPC commands0..1116 are replayed;
terminal BFM yaw4 executes1117..1416 and continuous separate250. No online MPC claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType

import mujoco
import numpy as np

from hybrid_checks import (SHAPES, arrays, exact, first_issue, flatten_history,
    history_snapshot, prefix_sample, quiet_metrics)

SWITCH, COUNT, EXTENSION, SPEC = 1117, 1417, 250, 8191

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    return value

def write(p, value):
    temp = p.with_suffix('.tmp.json')
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    os.replace(temp, p)

def integration(model, data):
    if int(mujoco.mjtState.mjSTATE_INTEGRATION) != SPEC or mujoco.mj_stateSize(model, SPEC) != 291:
        raise ValueError('native integration spec/size mismatch')
    value = np.empty(291, np.float64)
    mujoco.mj_getState(model, data, value, SPEC)
    return value

def physical_snapshot(model, data):
    return dict(integration=integration(model, data), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
        qacc_warmstart=data.qacc_warmstart.copy(), ctrl=data.ctrl.copy(), time=np.asarray(data.time),
        warning_number=data.warning.number.copy(), warning_lastinfo=data.warning.lastinfo.copy())

def verify_endpoint(endpoint, snapshot, count):
    if int(endpoint['completed_controls']) != count or int(endpoint['integration_state_spec']) != SPEC:
        raise ValueError('independent endpoint count/spec mismatch')
    exact(snapshot['integration'], endpoint['final_integration'], 'complete291 independent endpoint')
    for key in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'time', 'warning_number', 'warning_lastinfo'):
        exact(snapshot[key], endpoint['warning_counts' if key == 'warning_number' else key], 'independent endpoint ' + key)

def preflight(args):
    if mujoco.__version__ != '3.2.3' or np.__version__ != '1.26.4':
        raise ValueError('pinned native runtime mismatch')
    if (args.clip, args.switch_control, args.requested_controls, args.extension_controls) != ('walk002', SWITCH, COUNT, EXTENSION):
        raise ValueError('one fixed walk002 experiment only')
    receipt = json.loads(args.manifest.read_text())
    for path, digest in receipt['hashes'].items():
        if sha(path) != digest: raise ValueError('frozen input changed: ' + path)
    source = read(args.producer / 'trace.npz')
    report = json.loads((args.producer / 'report.json').read_text())
    request = json.loads((args.producer / 'request.json').read_text())
    audit = json.loads((args.physical_audit / 'report.json').read_text())
    intent = json.loads(args.intent_audit.read_text())
    endpoint = read(args.boundary_endpoint)
    endpoint_report = json.loads(args.boundary_endpoint.with_name('report.json').read_text())
    digest = sha(args.producer / 'trace.npz')
    if not (report['trace_sha256'] == digest and report['request_sha256'] == sha(args.producer / 'request.json')
            and report['failure'] is None and report['completed_controls'] == COUNT):
        raise ValueError('producer identity/completion mismatch')
    if not (audit['independent_segment_pass'] and audit['requested_controls'] == COUNT
            and audit['compared_physics_steps'] == COUNT*10 and digest in audit['input_hashes'].values()):
        raise ValueError('root physical qualification mismatch')
    if not (intent['full_lifecycle_source_intent_pass'] and not intent['requested_segment_quiet_pass']
            and intent['source_metrics']['source_controls'] == 667 and digest in intent['hashes'].values()):
        raise ValueError('root source/original quiet result mismatch')
    if not (endpoint_report['all_recorded_samples_bitexact'] and endpoint_report['completed_controls'] == SWITCH
            and endpoint_report['compared_physics_steps'] == SWITCH*10 and endpoint_report['integration_state_size'] == 291
            and endpoint_report['original_full_trace_controls'] == COUNT and endpoint_report['is_prefix_boundary']
            and endpoint_report['original_trace_sha256'] == digest and endpoint_report['endpoint_sha256'] == sha(args.boundary_endpoint)
            and str(endpoint['original_trace_sha256']) == digest):
        raise ValueError('independent switch endpoint provenance mismatch')
    if len(source['target']) != COUNT or not np.all(source['physics_substeps'] == 10):
        raise ValueError('source canonical count mismatch')
    return receipt, source, request, endpoint

class Hybrid:
    def __init__(self, args, receipt, source, request, endpoint):
        from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, motion_states
        from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed
        from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory
        from artifacts.teleop_six_hour_20260910.bfm_terminal_yaw4_goal import terminal_goal_yaw4
        self.args, self.receipt, self.source, self.request, self.endpoint = args, receipt, source, request, endpoint
        self.native, self.contract, original, timeline, manifest = load_native_bundle(args.bundle, args.clip)
        if manifest != request['model_manifest']: raise ValueError('model manifest mismatch')
        if not (self.native.nq, self.native.nv, self.native.nu, self.native.nbody) == (30, 29, 23, 25):
            raise ValueError('native topology mismatch')
        self.motion = read(args.reference)
        self.original29 = read(args.bundle / args.clip / 'original29.npz')
        for key, path in (('reference_sha256', args.reference), ('base_native_reference_sha256', args.bundle / args.clip / 'native_original.npz'),
                          ('original29_sha256', args.bundle / args.clip / 'original29.npz')):
            if sha(path) != request['motion_override'][key]: raise ValueError('reference identity mismatch: ' + key)
        phases = {p['name']: p for p in timeline['phases']}
        if not (timeline['total_requested_controls'] == COUNT and len(self.motion['joint_pos']) == 1428
                and phases['source_motion']['control_start'] == 350 and phases['source_motion']['control_stop'] == 1017
                and phases['return_ramp']['control_stop'] == SWITCH
                and phases['returned_standing']['requested_controls'] == 250
                and phases['standing_proof_margin']['requested_controls'] == 50):
            raise ValueError('original timeline mismatch')
        self.seed = Native23BFMRolloutSeed(self.native, self.contract, original, args.onnx, dependency_directory=args.dependencies, threads=1)
        self.seed._goal = MethodType(terminal_goal_yaw4, self.seed)
        self.history_type = BFMHistory
        self.controller_history, self.controller_previous, self.controller_count = None, None, None
        self.kp, self.kd, self.effort, self.speed = [np.asarray(self.contract[k]) for k in ('kp','kd','native_effort','native_velocity')]
        self.limits = self.native.jnt_range[1:]
        initial = motion_states(self.motion)[10]
        exact(initial[:30], source['qpos'][0], 'canonical qpos frame10')
        exact(initial[30:], source['qvel'][0], 'canonical qvel frame10')
        self.data = mujoco.MjData(self.native)
        self.data.qpos[:], self.data.qvel[:] = initial[:30], initial[30:]
        mujoco.mj_forward(self.native, self.data)
        self.expected_time, self.prefix_steps, self.history_checks = 0., 0, 0
        self.switch_verified = False
        self.started = time.perf_counter()
        self.initial_issue = self.native_issue()
        write(args.output / 'request.json', dict(kind='one_offline_saved_MPC_prefix_actual_BFM_terminal_hybrid', clip='walk002',
            requested_controls=COUNT, original_source_controls=667, switch_control=SWITCH, BFM_original_terminal_controls=300,
            separate_hold_controls=EXTENSION, original_main_trace_sha256=sha(args.producer/'trace.npz'),
            preceding_MPC_targets='saved actual targets0..1116, physically replayed from canonical v4 frame10',
            BFM_identity=self.seed.identity(), effective_terminal_yaw_gain=4., position_gain=1., goal_horizon=8,
            terminal_goal_reference=str(args.bundle/args.clip/'native_original.npz'), motion_override=request['motion_override'],
            active_terminal_history='separate controller_history/controller_previous/controller_count; seed remains prefix history after1117',
            physical_state_rewrites_after_initialization=0, root_assistance=False, private_rollouts=0,
            timing_or_live_qualified=False, hardware_authorized=False, manifest_sha256=sha(args.manifest)))

    def active(self):
        if self.controller_history is None:
            return self.seed.history, self.seed.previous_action, self.seed.recorded_controls
        return self.controller_history, self.controller_previous, self.controller_count

    def snapshot(self):
        h, p, n = self.active()
        return dict(**physical_snapshot(self.native, self.data), expected_time=np.asarray(self.expected_time),
                    **history_snapshot(h, p, n))

    def native_issue(self):
        if np.any(self.data.xfrc_applied) or np.any(self.data.qfrc_applied): return 'unexpected_external_force'
        return first_issue(self.data.qpos, self.data.qvel, self.data.qfrc_actuator[6:], self.data.warning.number,
            self.data.warning.lastinfo, float(self.data.time), self.expected_time, self.limits, self.speed, self.effort)

    def new_trace(self):
        trace = {key: [] for key in SHAPES}
        for key in ('qpos', 'physics_qpos'): trace[key].append(self.data.qpos.copy())
        for key in ('qvel', 'physics_qvel'): trace[key].append(self.data.qvel.copy())
        trace['physics_time'].append(float(self.data.time))
        trace['physics_expected_time'].append(self.expected_time)
        trace['physics_warning_number'].append(self.data.warning.number.copy())
        trace['physics_warning_lastinfo'].append(self.data.warning.lastinfo.copy())
        return trace

    def segment(self, start, stop, directory):
        initial = self.snapshot()
        trace, failure = self.new_trace(), None
        if self.initial_issue: failure = dict(kind=self.initial_issue, control=start, substep=0)
        for control in range(start, stop):
            if failure: break
            frame = min(control+11, len(self.motion['joint_pos'])-1)
            attempted = self.snapshot()
            try:
                if control == SWITCH:
                    if self.prefix_steps != SWITCH*10 or self.seed.recorded_controls != SWITCH:
                        raise ValueError('prefix count before switch mismatch')
                    verify_endpoint(self.endpoint, attempted, SWITCH)
                    exact(self.expected_time, self.endpoint['time'], 'independent accumulated switch clock')
                    self.controller_history = self.history_type()
                    for key in self.controller_history.data:
                        self.controller_history.data[key][:] = self.seed.history.data[key]
                    self.controller_previous = self.seed.previous_action.copy()
                    self.controller_count = self.seed.recorded_controls
                active_history, previous, active_count = self.active()
                if active_count != control: raise ValueError('active BFM history clock mismatch')
                previous = previous.copy()
                state, terms = self.seed._terms(self.data.qpos, self.data.qvel, previous)
                history = flatten_history(active_history)
                if control <= SWITCH:
                    exact(self.data.qpos, self.source['qpos'][control], 'prefix precontrol qpos')
                    exact(self.data.qvel, self.source['qvel'][control], 'prefix precontrol qvel')
                    exact(previous, self.source['fresh_seed_previous_action'][control], 'prefix prior action')
                    exact(history, self.source['fresh_seed_measured_history'][control], 'prefix measured300 history')
                    self.history_checks += 1
                tick = time.perf_counter()
                if control < SWITCH:
                    target = self.source['target'][control].copy()
                    exact(np.asarray(frame), self.source['source_frame'][control], 'prefix source frame')
                    self.seed.record_control(control, self.data.qpos, self.data.qvel, target)
                    action = self.seed.previous_action.copy()
                else:
                    if control == SWITCH:
                        np.savez_compressed(directory/'verified_switch_inputs.npz', **self.snapshot(), state=state, history=history,
                                            integration_state_spec=SPEC, original_trace_sha256=sha(self.args.producer/'trace.npz'))
                        write(directory/'verified_switch_receipt.json', dict(control=SWITCH, prefix_steps=self.prefix_steps,
                            history_checks=self.history_checks, complete291_endpoint_bitexact=True, state_copy_or_forward_at_boundary=False,
                            endpoint_sha256=sha(self.args.boundary_endpoint), first_BFM_command_executed=False))
                        self.switch_verified = True
                    if not self.switch_verified: raise ValueError('first BFM command without complete prefix gate')
                    pending_history = self.history_type()
                    for key in pending_history.data:
                        pending_history.data[key][:] = active_history.data[key]
                    exact(pending_history.before_update(terms), history, 'active terminal history update')
                    goal = self.seed._goal(frame, self.data.qpos)
                    raw = self.seed.sessions['actor'].run(None, dict(state=state[None], last_action=previous[None], history=history[None], z=goal))[0][0]
                    if raw.shape != (23,) or not np.isfinite(raw).all(): raise ValueError('nonfinite/malformed actor output')
                    action = raw*5
                    target = np.clip(np.asarray(self.contract['default_q'])+action*.25*np.asarray(self.contract['training_effort'])/self.kp,
                                     self.limits[:,0], self.limits[:,1])
                exact(integration(self.native, self.data), attempted['integration'], 'no physical mutation during control computation')
                if not np.isfinite(target).all(): raise ValueError('nonfinite target')
                if control >= SWITCH:
                    self.controller_history = pending_history
                    self.controller_previous = action.copy()
                    self.controller_count += 1
                inference_ms = (time.perf_counter()-tick)*1000
            except Exception as exc:
                failure = dict(kind='precontrol_gate_or_inference_failure', control=control, substep=0, exception_type=type(exc).__name__, message=str(exc))
                np.savez_compressed(directory/'failed_precontrol.npz', **attempted, integration_state_spec=SPEC)
                write(directory/'failed_precontrol.json', failure)
                break
            peak_range = peak_speed = peak_effort = 0.
            substeps = 0
            for sub in range(10):
                try:
                    issue = self.native_issue()
                    if issue: raise ValueError('before-step native gate: ' + issue)
                    torque = self.kp*(target-self.data.qpos[7:])-self.kd*self.data.qvel[6:]
                    if not np.isfinite(torque).all(): raise ValueError('nonfinite requested torque')
                    self.data.ctrl[:] = np.clip(torque, -self.effort, self.effort)
                    mujoco.mj_step(self.native, self.data)
                    self.expected_time += .002
                    substeps += 1
                    sample = dict(physics_qpos=self.data.qpos.copy(), physics_qvel=self.data.qvel.copy(),
                        physics_requested_torque=torque.copy(), physics_torque=self.data.ctrl.copy(), physics_actuator_force=self.data.qfrc_actuator[6:].copy(),
                        physics_time=np.asarray(float(self.data.time)), physics_expected_time=np.asarray(self.expected_time),
                        physics_warning_number=self.data.warning.number.copy(), physics_warning_lastinfo=self.data.warning.lastinfo.copy())
                    for key, value in sample.items(): trace[key].append(value)
                    peak_range = max(peak_range, float(np.maximum(self.limits[:,0]-self.data.qpos[7:], self.data.qpos[7:]-self.limits[:,1]).max()))
                    peak_speed = max(peak_speed, float(np.max(np.abs(self.data.qvel[6:])/self.speed)))
                    peak_effort = max(peak_effort, float(np.max(np.abs(self.data.qfrc_actuator[6:])/self.effort)))
                    if control < SWITCH:
                        prefix_sample(self.source, sample, control*10+sub+1)
                        self.prefix_steps += 1
                    issue = self.native_issue()
                    if issue: raise ValueError(issue)
                except Exception as exc:
                    failure = dict(kind='native_or_prefix_failure', control=control, substep=substeps,
                        exception_type=type(exc).__name__, message=str(exc), range_excess=peak_range,
                        velocity_ratio=peak_speed, effort_ratio=peak_effort)
                    break
            mujoco.mj_kinematics(self.native, self.data)
            values = dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(), target=target.copy(), source_frame=frame,
                global_control=control, controller_mode=int(control >= SWITCH), joint_error=self.data.qpos[7:]-self.motion['joint_pos'][frame],
                root_error=self.data.qpos[:3]-self.motion['body_pos_w'][frame,0], state=state, history=history, previous_action=previous,
                action=action.copy(), inference_ms=inference_ms, range_excess=peak_range, velocity_ratio=peak_speed,
                effort_ratio=peak_effort, physics_substeps=substeps)
            for key, value in values.items(): trace[key].append(value)
            if (control+1)%100 == 0 or failure:
                write(directory/'progress.json', dict(global_control=control+1, requested_global_stop=stop, physics_seconds=float(self.data.time), failure=failure))
                print(json.dumps(dict(global_control=control+1, failure=failure)), flush=True)
            if failure: break
        return self.save(directory, trace, failure, initial, start, stop)

    def save(self, directory, trace, failure, initial, start, stop):
        a, final = arrays(trace), self.snapshot()
        payload = dict(a, initial_integration=initial['integration'], final_integration=final['integration'], integration_state_spec=SPEC,
            **{'initial_'+key: value for key, value in initial.items() if key != 'integration'},
            **{'final_'+key: value for key, value in final.items() if key != 'integration'})
        np.savez_compressed(directory/'trace.npz', **payload)
        completed = int(np.sum(a['physics_substeps'] == 10))
        physical_pass = failure is None and completed == stop-start
        report = dict(kind='offline_recorded_MPC_prefix_actual_terminal_BFM' if start == 0 else 'continuous_actual_terminal_BFM_separate_hold',
            clip='walk002', global_start=start, global_stop_requested=stop, requested_controls=stop-start,
            completed_controls=completed, recorded_controls=len(a['target']), physics_steps=len(a['physics_torque']),
            completed=physical_pass, failure=failure, strict_physical_limits_pass=physical_pass,
            original_source_controls=667 if start == 0 and completed >= 1017 else 0,
            original_main_quiet_failure_preserved=True, quiet_last_three_seconds=quiet_metrics(a, self.original29),
            range_excess_max=float(np.max(a['range_excess'], initial=0)), velocity_ratio_max=float(np.max(a['velocity_ratio'], initial=0)),
            effort_ratio_max=float(np.max(a['effort_ratio'], initial=0)), engine_warning_counts=self.data.warning.number.tolist(),
            complete_prefix_physics_bitexact=self.prefix_steps == SWITCH*10, prefix_physics_steps=self.prefix_steps,
            measured_history_comparisons=self.history_checks, switch_verified=self.switch_verified,
            active_recorded_controls=int(final['recorded_controls']), physical_state_rewrites_after_initialization=0,
            root_assistance=False, timing_or_live_qualified=False, hardware_authorized=False,
            elapsed_wall_seconds=time.perf_counter()-self.started, trace_sha256=sha(directory/'trace.npz'),
            request_sha256=sha(self.args.output/'request.json'), manifest_sha256=sha(self.args.manifest))
        report['quiet_pass'] = bool(physical_pass and report['quiet_last_three_seconds']['pass_all'])
        write(directory/'report.json', report)
        return report, final

def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo))
    receipt, source, request, endpoint = preflight(args)
    engine = Hybrid(args, receipt, source, request, endpoint)
    main_report, boundary = engine.segment(0, COUNT, args.output)
    hold_report = None
    if main_report['strict_physical_limits_pass'] and main_report['completed_controls'] == COUNT:
        extension = args.output/'post_lifecycle_hold_5s'
        extension.mkdir()
        for key, value in boundary.items(): exact(value, engine.snapshot()[key], 'live main/hold continuity ' + key)
        np.savez_compressed(args.output/'boundary1417.npz', **boundary, integration_state_spec=SPEC)
        hold_report, final = engine.segment(COUNT, COUNT+EXTENSION, extension)
        extra = read(extension/'trace.npz')
        for key, value in boundary.items():
            exact(value, extra['initial_integration' if key == 'integration' else 'initial_'+key], 'saved complete main/hold continuity ' + key)
    hashes = {path: sha(path) == digest for path, digest in receipt['hashes'].items()}
    write(args.output/'postrun_provenance.json', dict(all_exact=all(hashes.values()), count=len(hashes), checks=hashes))
    summary = dict(kind='one_offline_walk002_terminal_BFM_hybrid', lifecycle=main_report, extension=hold_report,
        inputs_unchanged=all(hashes.values()), intended_lifecycle_controls=COUNT, intended_extension_controls=EXTENSION,
        original_main_quiet_failure_preserved=True, new_fresh_MPC_or_live_qualified=False, hardware_authorized=False)
    write(args.output/'comparison.json', summary)
    physical_complete = main_report['completed'] and hold_report is not None and hold_report['completed'] and all(hashes.values())
    return 0 if physical_complete and main_report['quiet_pass'] and hold_report['quiet_pass'] else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','bundle','reference','producer','physical-audit','intent-audit','boundary-endpoint','onnx','dependencies','manifest','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--switch-control', type=int, required=True)
    parser.add_argument('--requested-controls', type=int, required=True)
    parser.add_argument('--extension-controls', type=int, required=True)
    args = parser.parse_args()
    try:
        code = main(args)
    except Exception as exc:
        if args.output.exists(): write(args.output/'fatal.json', dict(kind='incomplete_exception', type=type(exc).__name__, message=str(exc)))
        raise
    sys.exit(code)
