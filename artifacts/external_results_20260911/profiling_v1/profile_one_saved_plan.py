"""Read-only, frozen-source single-plan profile; no executed controller steps."""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source_snapshot'))
import numpy as np
import mujoco
import mjbatch
from gear_sonic.utils import g1_true23_mjbatch_ilqr_core as core
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker, load_native_bundle, load_motion_override
from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed, BFMSeedRolloutError

BASE = Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
PRODUCER = BASE / 'mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1/final_frozen'
CHECKPOINT = BASE / 'mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1_checkpoint_03200.npz'
BUNDLE = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
ONNX = BUNDLE.parent / 'bfm_onnx_v2'
SEED = BUNDLE.parent / 'bfm_pico_feedback_v2/trace.npz'
REFERENCE = BASE / 'mjbatch_intent_floor_inputs_v1/pico/reference.npz'
DEPENDENCIES = Path('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_snapshot():
    return dict(monotonic_ns=time.perf_counter_ns(), load_average=os.getloadavg(),
                proc_stat=Path('/proc/stat').read_text(), cpuinfo=Path('/proc/cpuinfo').read_text(),
                affinity=sorted(os.sched_getaffinity(0)), process_threads=len(list(Path('/proc/self/task').iterdir())))


events, stack, captured = [], [], {}


def timed(name, function, capture=False):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        event = dict(name=name, parent=stack[-1] if stack else None, start_ns=time.perf_counter_ns())
        index = len(events); events.append(event); stack.append(index)
        try:
            result = function(*args, **kwargs)
        finally:
            event['end_ns'] = time.perf_counter_ns(); stack.pop()
        if capture:
            for j, value in enumerate(result if isinstance(result, tuple) else (result,)):
                if isinstance(value, np.ndarray):
                    # Keep returned arrays by reference; no mutation or extra numeric operation.
                    captured[f'{name.replace(".", "_")}_{index}_{j}'] = value
        return result
    return wrapped


class Proxy:
    def __init__(self, obj, name, methods):
        self._object = obj
        for method in methods:
            setattr(self, method, timed(name + '.' + method, getattr(obj, method)))
    def __getattr__(self, name):
        return getattr(self._object, name)


def main():
    start = time.perf_counter()
    request = json.loads((PRODUCER / 'request.json').read_text())
    with np.load(CHECKPOINT, allow_pickle=False) as f:
        checkpoint = {k: f[k].copy() for k in f.files}
    meta = json.loads(str(checkpoint['checkpoint_metadata']))
    completed = meta['completed_controls']
    assert completed == 3200 and checkpoint['checkpoint_warm_targets'].shape == (30,23)
    assert meta['request_sha256'] == sha(PRODUCER / 'request.json')
    with np.load(PRODUCER / 'trace.npz', allow_pickle=False) as f:
        truth = {k:f[k].copy() for k in ['qpos','qvel','target','planned_target','planned_state','feedback_gain','fresh_seed_previous_action','fresh_seed_measured_history']}
    np.testing.assert_array_equal(checkpoint['qpos'], truth['qpos'][:completed+1])
    np.testing.assert_array_equal(checkpoint['qvel'], truth['qvel'][:completed+1])
    native, contract, base_motion, timeline, manifest = load_native_bundle(BUNDLE, 'pico')
    motion, override = load_motion_override(REFERENCE, BUNDLE, 'pico', native, contract, base_motion, timeline, manifest)
    assert override['reference_sha256'] == request['motion_override']['reference_sha256']
    servo = position_servo_copy(native, contract['kp'], contract['kd'], contract['native_effort'])
    margin = request['costs']['all_joint_limit_override']
    planner = Native23Tracker(servo, contract, motion, horizon=30, threads=8,
        all_joint_limit_margin=margin['interior_margin_rad'], all_joint_limit_weight=margin['weight'],
        relative_foot_weight=request['costs']['root_relative_foot_position'], fd_epsilon=request['finite_difference_epsilon'])
    fresh_seed = Native23BFMRolloutSeed(native, contract, base_motion, ONNX, dependency_directory=DEPENDENCIES, threads=1)
    for control in range(completed):
        fresh_seed.record_control(control, truth['qpos'][control], truth['qvel'][control], truth['target'][control])
    flat_history = np.concatenate([fresh_seed.history.data[k].ravel() for k in sorted(fresh_seed.history.data)])
    np.testing.assert_array_equal(flat_history, truth['fresh_seed_measured_history'][completed])
    np.testing.assert_array_equal(fresh_seed.previous_action, truth['fresh_seed_previous_action'][completed])
    with np.load(SEED, allow_pickle=False) as f:
        seed_targets = f['target'].copy()
    assert sha(SEED) == request['recorded_target_seed']['trace_sha256']
    data = SimpleNamespace(qpos=truth['qpos'][completed].copy(), qvel=truth['qvel'][completed].copy())
    args = SimpleNamespace(horizon=30, commit=5, iterations=5)
    warm_targets = checkpoint['checkpoint_warm_targets'].copy()

    # Extract the existing producer's planning block, unchanged AST statements.
    runner_path = HERE / 'source_snapshot/runner_reference.py'
    tree = ast.parse(runner_path.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loop = next(n for n in run.body if isinstance(n, ast.While))
    first = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) == 'planner.window')
    last = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'solve_ms')
    selected = loop.body[first:last+1]
    signature = 'def one_plan(planner, fresh_seed, args, seed_targets, data, warm_targets, completed):\n    pass\n'
    fn = ast.parse(signature).body[0]
    fn.body = selected + ast.parse('return planned_states, planned_targets, gains, cost, seed_selection, solve_ms').body
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    generated = ast.unparse(module) + '\n'
    (HERE / 'extracted_planning_block.py').write_text(generated)
    assert [ast.dump(n, include_attributes=False) for n in ast.parse(generated).body[0].body[:-1]] == [ast.dump(n, include_attributes=False) for n in selected]

    # Instrument by delegation only. Frozen source files stay byte-identical.
    core.backward = timed('backward', core.backward, capture=True)
    for method in ['linearize','expand']:
        setattr(planner, method, timed(method, getattr(planner,method), capture=True))
    for method in ['advance','features','cost','step','probe']:
        setattr(planner, method, timed('planner.'+method, getattr(planner,method)))
    rollout = planner.rollout
    seed_names = iter(['recorded','shifted','fresh'])
    def classified_rollout(x0, us, gains=None):
        label = 'seed_scoring.' + next(seed_names) if gains is None else 'line_search_rollout'
        return timed(label, rollout, capture=True)(x0,us,gains)
    planner.rollout = classified_rollout
    planner.batch = Proxy(planner.batch, 'linear_batch', ['step','forward'])
    planner.line = Proxy(planner.line, 'line_batch', ['step'])
    fresh_seed.propose = timed('fresh_seed_generation', fresh_seed.propose, capture=True)
    fresh_seed._goal = timed('fresh_goal', fresh_seed._goal)
    for name in fresh_seed.sessions:
        fresh_seed.sessions[name] = Proxy(fresh_seed.sessions[name], 'onnx_'+name, ['run'])
    mujoco.mj_step = timed('fresh_native_step', mujoco.mj_step)
    scope = dict(np=np, time=time, ilqr=timed('ilqr',core.ilqr), BFMSeedRolloutError=BFMSeedRolloutError)
    exec(compile(module, str(HERE/'extracted_planning_block.py'), 'exec'), scope)
    before = cpu_snapshot()
    result = timed('planning_total', scope['one_plan'])(planner,fresh_seed,args,seed_targets,data,warm_targets,completed)
    after = cpu_snapshot()
    xs, us, gains, cost, selection, original_solve_ms = result
    np.testing.assert_array_equal(flat_history, np.concatenate([fresh_seed.history.data[k].ravel() for k in sorted(fresh_seed.history.data)]))
    np.testing.assert_array_equal(fresh_seed.previous_action, truth['fresh_seed_previous_action'][completed])
    plan = next(p for p in json.loads((PRODUCER/'plans.json').read_text()) if p['control'] == completed)
    comparisons = {}
    for name, actual, expected in [('planned_state',xs[:5],truth['planned_state'][completed:completed+5]),('planned_target',us[:5],truth['planned_target'][completed:completed+5]),('feedback_gain',gains[:5],truth['feedback_gain'][completed:completed+5])]:
        comparisons[name] = dict(bit_exact=bool(np.array_equal(actual,expected)), maximum_absolute_error=float(np.max(np.abs(actual-expected))))
    comparisons['cost'] = dict(bit_exact=bool(float(cost)==plan['cost']), actual=float(cost), saved=plan['cost'])
    captured.update(actual_state=np.r_[data.qpos,data.qvel], checkpoint_warm_targets=warm_targets,
                    history=flat_history, previous_action=fresh_seed.previous_action,
                    planned_states=xs, planned_targets=us, gains=gains, cost=np.asarray(cost))
    tick=time.perf_counter(); np.savez_compressed(HERE/'solve_arrays.npz',**captured)
    array_serialization_seconds=time.perf_counter()-tick
    # Separate exact existing checkpoint serialization workload; outside solve timing.
    spec=importlib.util.spec_from_file_location('frozen_runner',runner_path)
    frozen=importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    tick=time.perf_counter(); frozen.atomic_trace(HERE/'checkpoint_serialization_sample.npz',checkpoint)
    checkpoint_serialization_seconds=time.perf_counter()-tick
    with np.load(HERE/'checkpoint_serialization_sample.npz',allow_pickle=False) as f:
        for k in checkpoint: np.testing.assert_array_equal(f[k],checkpoint[k],err_msg=k)
    for index,event in enumerate(events):
        event['inclusive_ms']=(event['end_ns']-event['start_ns'])/1e6
        event['exclusive_ms']=event['inclusive_ms']-sum((child['end_ns']-child['start_ns'])/1e6 for child in events if child['parent']==index)
    totals={}
    for event in events:
        row=totals.setdefault(event['name'],dict(calls=0,inclusive_ms=0.,exclusive_ms=0.))
        row['calls']+=1
        for key in ['inclusive_ms','exclusive_ms']:row[key]+=event[key]
    relevant=[p for p in (HERE/'source_snapshot').rglob('*.py')]
    binaries=[Path(mujoco.__file__),Path(mjbatch.__file__)]+list(Path(mjbatch.__file__).parent.glob('*.so'))
    report=dict(kind='single_frozen_native23_planning_profile', control=completed, source_state_frame=completed+10,
        same_producer_planning_statements_ast_exact=True, horizon=30, iterations=5, threads=8, commit=5,
        measured_input_history_bit_exact=True, original_seed_selection=plan['seed_selection'], seed_selection=selection,
        saved_producer_output_comparisons=comparisons, original_solve_ms=float(original_solve_ms), stage_times=totals,
        artifact_array_serialization_seconds=array_serialization_seconds,
        separate_checkpoint_serialization_seconds=checkpoint_serialization_seconds,
        checkpoint_serialization_payload_controls=completed, checkpoint_serialization_array_equality=True,
        serialization_in_solver_timer=False, total_process_seconds=time.perf_counter()-start,
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,mujoco=mujoco.__version__,platform=platform.platform(),
                     cpu_count=os.cpu_count(),environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),
        cpu_before=before,cpu_after=after, frozen_sources={str(p):sha(p) for p in relevant},binary_hashes={str(p):sha(p) for p in binaries},
        inputs={str(p):sha(p) for p in [CHECKPOINT,PRODUCER/'trace.npz',PRODUCER/'request.json',REFERENCE,SEED,BUNDLE/'manifest.json']},
        outputs={str(HERE/'solve_arrays.npz'):sha(HERE/'solve_arrays.npz')},
        profile_script_sha256=sha(__file__), no_controller_execution=True,no_hardware_interface=True,
        timing_scope='One cold-start process, exact saved planning input; shared host, no statistical timing qualification.')
    (HERE/'events.json').write_text(json.dumps(events,indent=2))
    (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(control=completed,timing=totals,comparison=comparisons,checkpoint_serialization_seconds=checkpoint_serialization_seconds)),flush=True)


if __name__=='__main__':
    main()
