"""One frozen 364-case comparison; all private computed suffixes retained."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
sys.path.insert(0, str(BASE / 'preserved_walk_demo_v1/repo'))
import numpy as np
import mujoco
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle
from bulk_forecast import BulkForecast

spec = importlib.util.spec_from_file_location('independent_oracle', BASE / 'phase_student_preallocated_forecast_v2/oracle_snapshot.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def path(p):
    s = str(p).replace('\\', '/')
    return Path('/mnt/' + s[0].lower() + s[2:]) if len(s)>2 and s[1]==':' else Path(s)


def exact(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), name


def main():
    request = json.loads((HERE / 'request.json').read_text())
    assert request['case_count'] == 364 and request['connected_controller'] is False
    for p, expected in request['input_sha256'].items():
        assert sha(path(p)) == expected, p
    model, contract, *_ = load_native_bundle(ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1', 'walk003')
    engine = BulkForecast(model, contract)
    source = mujoco.MjData(model)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before, after, state = np.empty(291), np.empty(291), np.empty(291)
    shape = dict(physics_qpos=(51,30), physics_qvel=(51,29), physics_time=(51,),
                 physics_torque=(50,23), physics_actuator_force=(50,23), warning_counts=(51,8), warning_lastinfo=(51,8))
    storage = {name: {k: np.zeros((364,*s), dtype=np.int32 if k.startswith('warning') else np.float64)
                     for k,s in shape.items()} for name in ('bulk','oracle')}
    frozen = [json.loads((BASE / 'phase_student_preallocated_forecast_v2/results/report.json').read_text())['rows'],
              json.loads((BASE / 'filtered_all_candidates132_v1/results/report.json').read_text())['rows']]
    output = HERE / 'results'
    output.mkdir(exist_ok=False)
    rows = []
    index = 0
    try:
        for set_index, definition in enumerate(request['case_sets']):
            with np.load(definition['path'], allow_pickle=False) as a:
                cases = {k:a[k].copy() for k in a.files}
            for local in range(definition['count']):
                np.copyto(state, cases['integration'][local])
                mujoco.mj_setState(model, source, state, state_spec)
                source.warning.number[:] = cases['warning_counts'][local]
                source.warning.lastinfo[:] = cases['warning_lastinfo'][local]
                mujoco.mj_forward(model, source)
                mujoco.mj_setState(model, source, state, state_spec)
                source.warning.number[:] = cases['warning_counts'][local]
                source.warning.lastinfo[:] = cases['warning_lastinfo'][local]
                mujoco.mj_getState(model, source, before, state_spec)
                exact(before, state, 'source full291')
                exact(source.qpos, cases['qpos'][local], 'source qpos')
                exact(source.qvel, cases['qvel'][local], 'source qvel')
                controls = int(cases['horizon_controls'][local]) if 'horizon_controls' in cases else 5
                target = cases['target'][local]
                targets = np.tile(target, (controls,1))
                results, timings = {}, {}
                order = ('bulk','oracle') if index%2 == 0 else ('oracle','bulk')
                for name in order:
                    tick = time.perf_counter_ns()
                    report, trace = engine.predict(source,target,controls) if name=='bulk' else oracle.inspect_native_segment(
                        model,source,targets,contract,stop_on_failure=False,retain_trace=True)
                    for k,a in trace.items():
                        storage[name][k][index,:len(a)] = a
                    timings[name] = (time.perf_counter_ns()-tick)/1e6
                    results[name] = report, trace
                    mujoco.mj_getState(model,source,after,state_spec)
                    exact(before,after,'caller full291 unchanged')
                    exact(source.warning.number,cases['warning_counts'][local],'caller warnings')
                    exact(source.warning.lastinfo,cases['warning_lastinfo'][local],'caller warning info')
                a,b = results['bulk'], results['oracle']
                for k in shape:
                    exact(a[1][k],b[1][k],f'case {index} full computed {k}')
                for key in ('feasible','first_failure','physics_steps','maximum','minimum_root_height_m',
                            'initial_time','final_time','final_warning_counts','original_data_unchanged'):
                    assert a[0][key] == b[0][key], (index,key,a[0][key],b[0][key])
                old = frozen[set_index][local]
                assert a[0]['first_failure'] == old['first_failure']
                assert a[0]['feasible'] == old['feasible']
                assert a[0]['checked_prefix_steps'] == old['physics_steps']
                if 'maximum' in old:
                    assert a[0]['prefix_maximum'] == old['maximum']
                row = dict(case=index, case_set=set_index, local_case=local,
                           control=int(cases['control'][local]), horizon_controls=controls,
                           all_seven_computed_fields_bitexact=True, all_report_fields_exact=True,
                           saved_first_failure_and_prefix_steps_exact=True, caller_unchanged=True,
                           bulk=a[0], timing_ms=timings, timing_order=order)
                rows.append(row)
                index += 1
                if index%32 == 0:
                    print(json.dumps(dict(completed=index,latest_ms=timings)),flush=True)
    except Exception as error:
        np.savez_compressed(output/'partial_traces.npz', **{name+'_'+k:a[:index+1] for name, fields in storage.items() for k,a in fields.items()})
        (output/'failure.json').write_text(json.dumps(dict(completed=index, exception=type(error).__name__, message=str(error), rows=rows),indent=2)+'\n')
        raise
    assert index == engine.forecasts == 364
    for name, fields in storage.items():
        np.savez_compressed(output/(name+'_traces.npz'), **fields,
                            computed_steps=np.array([r['bulk']['computed_steps'] for r in rows]),
                            checked_prefix_steps=np.array([r['bulk']['checked_prefix_steps'] for r in rows]))
    summary = {}
    for controls in (1,5):
        subset = [r for r in rows if r['horizon_controls'] == controls]
        summary[str(controls)] = {name:np.percentile([r['timing_ms'][name] for r in subset],[50,95,100]).tolist() for name in ('bulk','oracle')}
        summary[str(controls)]['native_loop'] = np.percentile([r['bulk']['native_rollout_ms'] for r in subset],[50,95,100]).tolist()
        summary[str(controls)]['assessment'] = np.percentile([r['bulk']['assessment_ms'] for r in subset],[50,95,100]).tolist()
    for p, expected in request['input_sha256'].items():
        assert sha(path(p)) == expected, p
    result = dict(kind='fixed364_private_native_bulk_parity', pass_=True, completed_cases=364,
                  computed_steps=sum(r['bulk']['computed_steps'] for r in rows),
                  checked_prefix_steps=sum(r['bulk']['checked_prefix_steps'] for r in rows),
                  computed_after_first_failure=sum(r['bulk']['computed_steps_after_first_failure'] for r in rows),
                  all_computed_trace_fields_bitexact=True, all_first_failures_exact=True,
                  native_library_version=323, strict_early_stop=False, actual_plant_steps=0,
                  timing_ms_p50_p95_max=summary, timing_includes='restore-forward-restore, native loop, unchanged scalar NumPy strict predicates, report and output storage copy',
                  contention_uncontrolled=True, real_time_qualified=False, connected_controller=False,
                  request_sha256=sha(HERE/'request.json'),
                  trace_sha256={name:sha(output/(name+'_traces.npz')) for name in storage}, rows=rows)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('pass_','completed_cases','computed_steps','checked_prefix_steps','computed_after_first_failure','timing_ms_p50_p95_max')}),flush=True)


if __name__ == '__main__':
    main()
