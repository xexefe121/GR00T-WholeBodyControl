"""One selected actual-state baseline collection; no optimizer or physics."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from collection_checks import (KEYS, SHAPES, archive, arrays_from_rows, atomic_write,
    check_frozen, exact, local, measured_rows, phase_code, read, sha, validate_snapshots,
    validate_trace, write)


class CollectionState:
    def __init__(self):
        self.rows = {key: [] for key in SHAPES}
        self.named = {key: [] for key in KEYS}
        self.inference_attempts = 0
        self.current = None

    def collect(self, stream, trace, phases, start, stop, infer, features, progress):
        for row in stream:
            control = row['control']
            self.current = row
            if control < start or control >= stop:
                continue
            target = trace['target'][control]
            self.inference_attempts += 1
            raw, base, sensed = infer(row, control + 11)
            exact(sensed, row['state'], 'baseline sensor input')
            assert raw.shape == base.shape == target.shape == (23,)
            assert np.isfinite(raw).all() and np.isfinite(base).all()
            x = features(row['qpos'], row['qvel'], control + 11, base, row['previous_action'])
            assert x.shape == (1069,) and x.dtype == np.float32 and np.isfinite(x).all()
            values = dict(features=x, residual_rad=target-base, base_target=base,
                          expert_target=target.copy(), base_action=raw,
                          previous_action=row['previous_action'].copy(), history=row['history'].copy(),
                          state=sensed, source_frame=control+11, control=control,
                          phase=phase_code(control, phases))
            assert all(np.isfinite(value).all() for value in values.values())
            for key, value in values.items():
                self.rows[key].append(value)
            for key in KEYS:
                self.named[key].append(row['named_history'][key].copy())
            progress(len(self.rows['control']), control)
        assert len(self.rows['control']) == stop - start
        assert self.inference_attempts == stop - start
        return arrays_from_rows(self.rows)


def save_rows(path, state, trace, snapshots, contract, complete):
    arrays = arrays_from_rows(state.rows)
    n = len(arrays['control'])
    exact(arrays['control'], np.arange(250, 250+n, dtype=np.int64), 'collected contiguous controls')
    limits = np.asarray(contract['joint_limits'], np.float64)
    span = np.diff(limits, axis=1).ravel().astype(np.float32)
    for key in ('expert_target',):
        exact(arrays[key], trace['target'][250:250+n], 'actual native target labels')
    assert np.all(arrays['expert_target'] >= limits[:, 0]) and np.all(arrays['expert_target'] <= limits[:, 1])
    exact(arrays['residual_rad'], arrays['expert_target']-arrays['base_target'], 'unclipped residual')
    dimensions = dict(actions=23, base_ang_vel=3, dof_pos=23, dof_vel=23, projected_gravity=3)
    named = {'history_'+key: np.asarray(state.named[key], np.float32).reshape(n, 4, dimensions[key]) for key in KEYS}
    np.savez_compressed(path, **arrays, **named, joint_limits=limits, joint_span=span,
        teacher_qpos=trace['qpos'][250:251+n], teacher_qvel=trace['qvel'][250:251+n],
        control_integration_before=snapshots['control_integration_before'][250:250+n],
        integration_spec=snapshots['integration_spec'],
        control_time_before=snapshots['time'][250:250+n],
        control_warning_counts_before=snapshots['warning_counts'][250:250+n],
        control_warning_lastinfo_before=snapshots['warning_lastinfo'][250:250+n],
        complete=np.asarray(complete), original_trace_sha256=snapshots['original_trace_sha256'])
    return arrays


def preserved_failure(path, state, snapshots, phase, moving_stop):
    if state.current is None:
        return
    row = state.current
    control = row['control']
    np.savez_compressed(path, control=np.asarray(control), qpos=row['qpos'], qvel=row['qvel'],
        state=row['state'], previous_action=row['previous_action'], history=row['history'],
        failure_phase=np.asarray(phase), selected_moving_control=np.asarray(250 <= control < moving_stop),
        control_integration_before=snapshots['control_integration_before'][control],
        integration_spec=snapshots['integration_spec'], control_time_before=snapshots['time'][control],
        control_warning_counts_before=snapshots['warning_counts'][control],
        control_warning_lastinfo_before=snapshots['warning_lastinfo'][control],
        original_trace_sha256=snapshots['original_trace_sha256'],
        **{'history_'+key: value for key, value in row['named_history'].items()})


def preflight(manifest):
    contract = read(local(manifest['bundle']) / 'contract.json')
    prepared = []
    for case in manifest['cases']:
        trace, phases, qualification, proof = validate_trace(case, contract)
        snapshots, snapshot_proof = validate_snapshots(case, trace, qualification)
        proof['full_integration'] = snapshot_proof
        prepared.append((case, trace, snapshots, phases, proof))
    norm = archive(local(manifest['normalization']))
    assert norm['feature_mean'].shape == norm['feature_std'].shape == (1069,)
    assert np.isfinite(norm['feature_mean']).all() and np.isfinite(norm['feature_std']).all()
    assert np.all(norm['feature_std'] > 0)
    limits = np.asarray(contract['joint_limits'], np.float64)
    span = np.diff(limits, axis=1).ravel().astype(np.float32)
    for path in manifest['existing_labels'].values():
        with np.load(local(path), allow_pickle=False) as data:
            exact(data['joint_span'], span, 'unchanged original joint span')
            exact(data['joint_limits'], limits, 'unchanged original native limits')
    return prepared, contract, norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    manifest = check_frozen(args.manifest)
    prepared, contract, normalization = preflight(manifest)
    preflight_report = dict(kind='two_clip_saved_array_collection_preflight', pass_all=True,
        manifest_sha256=sha(args.manifest), cases={entry[0]['clip']: entry[4] for entry in prepared},
        selected_rows=6847, inference_calls=0, physics_steps=0, model_constructed=False)
    if args.preflight_only:
        write(args.manifest.parent / ('preflight_frozen.json' if manifest['frozen'] else 'preflight_draft.json'), preflight_report)
        print('Saved-array preflight passed: 6847 selected rows, no inference or dynamics.', flush=True)
        return
    assert manifest['frozen'] is True, 'Collection requires the final frozen manifest.'
    destination = args.manifest.parent / 'collection'
    destination.mkdir(exist_ok=False)
    write(destination / 'preflight.json', preflight_report)
    write(destination / 'request.json', dict(manifest_sha256=sha(args.manifest), selected_rows=6847,
        cases=manifest['cases'], baseline_goal=manifest['baseline_goal'], fitting_authorized=False,
        normalization_refitted=False, collector_physics_steps=0))
    started = time.perf_counter()
    results = {}
    active = None
    active_data = None
    state = None
    phase = 'before_model_import'
    try:
        import mujoco
        assert mujoco.__version__ == '3.2.3' and np.__version__ == '1.26.4'
        def forbidden_physics(*args, **kwargs):
            raise RuntimeError('Physics execution is forbidden in this collector.')
        for name in ('mj_step', 'mj_step1', 'mj_step2'):
            setattr(mujoco, name, forbidden_physics)
        from student_linear_runtime import LinearFeatures, infer_base, Native23BFMRolloutSeed, OFFSETS
        from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, load_motion_override
        source = Path(__file__).resolve().parent
        assert str(source) in str(Path(sys.modules['student_linear_runtime'].__file__).resolve())
        exact(OFFSETS, np.array([0, 1, 2, 4, 8, 16, 24, 37], np.int64), 'received offsets')
        imports = {}
        for name, module in list(sys.modules.items()):
            if name.startswith('gear_sonic') and getattr(module, '__file__', None):
                p = Path(module.__file__).resolve()
                assert p.is_relative_to(source), 'Unfrozen module import: ' + str(p)
                imports[name] = dict(path=str(p), sha256=sha(p))
        write(destination / 'frozen_imports.json', imports)
        np.savez_compressed(destination / 'existing_normalization.npz',
            feature_mean=normalization['feature_mean'], feature_std=normalization['feature_std'])
        for case, trace, snapshots, phases, proof in prepared:
            active = case['clip']
            active_data = (trace, snapshots)
            folder = destination / active
            folder.mkdir(exist_ok=False)
            state = CollectionState()
            phase = 'construct_unchanged_native_and_BFM_baseline'
            native, c, original, timeline, bundle_manifest = load_native_bundle(local(manifest['bundle']), active)
            motion, _ = load_motion_override(local(case['reference']), local(manifest['bundle']), active,
                native, c, original, timeline, bundle_manifest)
            assert c == contract
            original29 = archive(local(case['original29']))
            feature_builder = LinearFeatures(motion, original29, c)
            seed = Native23BFMRolloutSeed(native, c, original, local(manifest['onnx']),
                dependency_directory=local(manifest['onnx_dependencies']), threads=1)
            identity = seed.identity()
            assert identity['original_goal_horizon'] == 8 and identity['position_gain'] == 1. and identity['yaw_gain'] == 2.
            for path, digest in identity['runtime_binary_sha256'].items():
                assert digest in manifest['input_hashes'].values(), 'Unpinned actor runtime: ' + path
            write(folder / 'seed_identity.json', identity)
            phase = 'actual_state_baseline_collection'
            def infer(row, frame):
                return infer_base(seed, row['qpos'], row['qvel'], row['previous_action'], row['history'], frame)
            def progress(count, control):
                if count == 1 or count % 100 == 0 or count == case['selected_rows']:
                    atomic_write(destination / 'progress.json', dict(clip=active, completed_rows_this_clip=count,
                        completed_rows_total=sum(r['samples'] for r in results.values())+count,
                        selected_rows_total=6847, control=control, source_frame=control+11,
                        inference_attempts_this_clip=state.inference_attempts, physics_steps=0,
                        elapsed_seconds=time.perf_counter()-started))
            stream = measured_rows(trace, c, case['history_key'], case['previous_key'], case['moving_stop'])
            state.collect(stream, trace, phases, 250, case['moving_stop'], infer, feature_builder, progress)
            arrays = save_rows(folder / 'labels.npz', state, trace, snapshots, c, True)
            assert len(arrays['control']) == case['selected_rows']
            exact(arrays['source_frame'], arrays['control']+11, 'collected source frames')
            result = dict(kind='qualified_actual_moving_native_target_residual_labels', clip=active,
                complete=True, samples=len(arrays['control']), controls_inclusive=[250, case['moving_stop']-1],
                phase_counts=np.bincount(arrays['phase'], minlength=3).tolist(),
                labels_sha256=sha(folder / 'labels.npz'), source_trace_sha256=sha(local(case['trace'])),
                root_qualification_sha256=sha(local(case['qualification'])),
                root_snapshots_sha256=sha(local(case['snapshots'])), root_snapshot_report_sha256=sha(local(case['snapshot_report'])),
                manifest_sha256=sha(args.manifest), observation_preflight=proof,
                features=1069, actor_inference_calls=state.inference_attempts,
                backward_inference_calls=state.inference_attempts, baseline_unclipped=True,
                baseline_goal='original_h8_position1_yaw2', terminal_rows_excluded=True,
                actual_native_targets_unchanged=True, full291_snapshots_from_root_only=True,
                reconstruction_max_abs_rad=float(np.max(np.abs(arrays['base_target']+arrays['residual_rad']-arrays['expert_target']))),
                previous_action_max_abs=float(np.max(np.abs(arrays['previous_action']))),
                previous_action_components_outside_five=int(np.sum(np.abs(arrays['previous_action']) > 5)),
                normalization_refitted=False, physics_steps=0, optimizer_calls=0, fitting_launched=False)
            write(folder / 'report.json', result)
            results[active] = result
        phase = 'saved_array_compatibility'
        assert sum(result['samples'] for result in results.values()) == 6847
        write(destination / 'rows_complete.json', dict(complete=True, completed_rows=6847,
            manifest_sha256=sha(args.manifest),
            case_report_sha256={clip:sha(destination/clip/'report.json') for clip in results},
            inference_only=True, physics_steps=0, fitting_launched=False))
        from audit_collection_compatibility import run as audit_compatibility
        compatibility = audit_compatibility(manifest, destination, sha(args.manifest))
        phase = 'final_hash_recheck'
        check_frozen(args.manifest)
        assert sum(result['samples'] for result in results.values()) == 6847
        write(destination / 'report.json', dict(kind='one_selected_PICO_walk002_collection', complete=True,
            selected_rows=6847, completed_rows=6847, cases=results, all_inputs_unchanged=True,
            manifest_sha256=sha(args.manifest), actor_inference_calls=6847, backward_inference_calls=6847,
            feature_count=1069, normalization_refitted=False, walk008_excluded=True,
            bounded_compatibility_report_sha256=sha(destination/'compatibility/report.json'),
            exact_target_conflict_groups=compatibility['exact_target_conflict_groups'],
            physics_steps=0, optimizer_calls=0, fitting_launched=False,
            elapsed_seconds=time.perf_counter()-started))
        print('Collection complete: PICO5980 + walk002867 actual moving rows, no fitting or dynamics.', flush=True)
    except BaseException as error:
        failure = dict(kind='collection_failure', complete=False, phase=phase, active_clip=active,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            completed_clips=results, selected_rows=6847, physics_steps=0, fitting_launched=False)
        if state is not None and active_data is not None:
            folder = destination / active
            preserved_failure(folder / 'failed_row_context.npz', state, active_data[1], phase, case['moving_stop'])
            try:
                save_rows(folder / 'partial_labels.npz', state, *active_data, contract, False)
            except BaseException as save_error:
                failure['partial_save_error'] = repr(save_error)
            failure['inference_attempts_this_clip'] = state.inference_attempts
            failure['completed_rows_this_clip'] = len(state.rows['control'])
        write(destination / 'failure.json', failure)
        raise


if __name__ == '__main__':
    main()
