"""Saved-array comparison only. Does not read files or execute a controller."""
import numpy as np


def exact(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise ValueError('Different actual boundary: ' + name)


def one_row(values, name, control=251):
    matches = np.flatnonzero(np.asarray(values[name]) == control)
    if len(matches) != 1:
        raise ValueError('Expected one prespecified control251: ' + name)
    return int(matches[0])


def error(a, b):
    difference = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if difference.shape != (23,) or not np.isfinite(difference).all():
        raise ValueError('Finite23 target difference required')
    return dict(rmse_rad=float(np.sqrt(np.mean(difference**2))),
                max_abs_rad=float(np.max(np.abs(difference))),
                difference_by_joint_rad=difference.tolist())


def compare(snapshot, first, actual_rows, fixed_maps, branch):
    """Compare first fresh proposal and saved applied student/fixed-map targets.

    Caller binds all files and completed owner separately. This function never
    grants label admission and never assumes the fresh proposal was applied.
    """
    row = one_row(actual_rows, 'control')
    old = one_row(fixed_maps, 'control')
    if int(first['global_control']) != 251 or not bool(first['proposal_before_preview']):
        raise ValueError('Wrong first fresh optimized proposal')
    exact(first['actual_state'], np.r_[snapshot['qpos'], snapshot['qvel']], 'fresh proposal state')
    exact(first['incoming_prior'], snapshot['previous_action'], 'fresh proposal prior')
    exact(first['incoming_history'], snapshot['history_flat'], 'fresh proposal history')
    for name in ('qpos', 'qvel'):
        exact(actual_rows[name][row], snapshot[name], 'student ' + name)
    exact(actual_rows['previous_action'][row], snapshot['previous_action'], 'student prior')
    exact(actual_rows['history'][row], snapshot['history_flat'], 'student history')
    exact(actual_rows['target'][row], fixed_maps['head_target'][old], 'saved student target aliases')
    matches = np.flatnonzero(branch['global_control'] == 251)
    if len(matches) > 1:
        raise ValueError('Repeated recovery control251')
    issued = len(matches) == 1
    steps = 0
    if issued:
        index = int(matches[0])
        exact(branch['target'][index], first['target'], 'issued fresh target')
        exact(branch['control_integration_before'][index], snapshot['integration'], 'issued full291')
        steps = int(branch['physics_substeps'][index])
        if not 0 <= steps <= 10:
            raise ValueError('Invalid actual first-control substeps')
    return dict(control=251, same_actual_state_prior_history=True,
                fresh_minus_fixed_map=error(first['target'], fixed_maps['map_target'][old]),
                fresh_minus_student=error(first['target'], actual_rows['target'][row]),
                student_minus_fixed_map=error(actual_rows['target'][row], fixed_maps['map_target'][old]),
                fresh_target_was_issued=issued, first_control_native_steps=steps,
                first_control_completed=steps == 10,
                old_label_kind='same-clock frozen prior expert feedback map; no fresh replan',
                new_target_kind='single selected fresh expert optimization from actual pre251',
                labels_admissible=False, model_calls=0, native_steps_executed=0,
                recovery_qualification_inferred=False)
