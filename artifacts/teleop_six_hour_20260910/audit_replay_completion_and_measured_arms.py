"""Read-only witnesses for replay completion and measured-body arm IK semantics.

No controller integration, physical stepping, or source-file edits. Completion
expressions are extracted from the actual replay runner rather than recopied.
The arm translation test uses FK/IK only; it cannot qualify physical tracking.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA, MODEL
from gear_sonic.utils.g1_true23_intent_arm_ik import Native23ArmIK


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_witness():
    path = ROOT / 'gear_sonic/scripts/evaluate_g1_true23_mjbatch_plan_replay.py'
    tree = ast.parse(path.read_text())
    result = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'result' for target in node.targets)
                  and isinstance(node.value, ast.Call)
                  and any(keyword.arg == 'full_source_completed' for keyword in node.value.keywords))
    names = ('completed_full_controls', 'full_source_completed', 'full_lifecycle_completed', 'plan_replayed')
    expressions = {kw.arg: kw.value for kw in result.keywords if kw.arg in names}
    assert set(expressions) == set(names)
    cases = []
    for name, steps, source_stop, lifecycle_count, failure, expected in (
        ('terminal_source_slot_partial_without_new_failure', [10] * 9 + [7], 10, 10, None, (False, False)),
        ('same_source_and_lifecycle_all_slots_full', [10] * 10, 10, 10, None, (True, True)),
        ('full_source_then_partial_final_recovery', [10] * 9 + [7], 8, 10, None, (True, False)),
        ('full_slots_but_new_physical_failure', [10] * 10, 10, 10, {'kind': 'physical_limit'}, (False, False)),
        ('prefix_short_of_source', [10] * 7, 10, 10, None, (False, False)),
    ):
        scope = dict(np=np, full_slots=np.asarray(steps) == 10, count=len(steps),
                     phase={'control_stop': source_stop},
                     motion={'joint_pos': np.zeros((lifecycle_count + 11, 23))},
                     plan={'target': np.zeros((len(steps), 23))}, failure=failure)
        actual = {key: eval(compile(ast.Expression(expr), str(path), 'eval'), scope)
                  for key, expr in expressions.items()}
        assert (actual['full_source_completed'], actual['full_lifecycle_completed']) == expected
        assert actual['completed_full_controls'] == sum(step == 10 for step in steps)
        cases.append(dict(name=name, physics_substeps=steps, failure=failure, result=actual,
                          source_control_stop=source_stop, lifecycle_count=lifecycle_count))
    return dict(runner=str(path), runner_sha256=sha(path),
                expressions={name: ast.unparse(expr) for name, expr in expressions.items()},
                cases=cases, all_passed=True,
                interpretation='Replaying all available plan slots can be true for a partial plan; full-source and full-lifecycle flags must remain false at their partial terminal slot.')


def arm_witness():
    model_path = ROOT.parent / 'GR00T-WholeBodyControl' / MODEL
    model = mujoco.MjModel.from_xml_path(str(model_path))
    run = BASE / 'bfm_measured_arm_ik_ff_v1/walk002'
    with np.load(run / 'trace.npz', allow_pickle=False) as z:
        trace = {key: z[key].copy() for key in ('qpos', 'ik_target', 'target')}
    original_path = DATA / 'walk002/original_source_bundle_v1/original_reference.npz'
    with np.load(original_path, allow_pickle=False) as z:
        original_root = z['source_qpos29'][:, :3].copy()
        positions = z['source_task_position_w'].copy()
        quaternions = z['source_task_quaternion_wxyz'].copy()
    frame_control = 420
    frame = frame_control + 11
    seed = trace['qpos'][frame_control].copy()
    previous = seed.copy()
    previous[20:] = trace['ik_target'][frame_control - 1]
    seed[20:] = previous[20:]
    goal = seed[:3] + positions[frame, :2] - original_root[frame]
    solver = Native23ArmIK(model, orientation_weight=.03, posture_weight=.02, max_nfev=6)
    reference = solver.solve(seed, goal, quaternions[frame, :2], posture_qpos=seed,
                             previous_qpos=previous, max_step_rad=.12)
    checks = []
    for translation in (np.array((0., 0., 0.)) - seed[:3], np.array((4., -3., 2.)), np.array((-10., 5., -.8))):
        shifted_seed, shifted_previous = seed.copy(), previous.copy()
        shifted_seed[:3] += translation
        shifted_previous[:3] += translation
        shifted = solver.solve(shifted_seed, goal + translation, quaternions[frame, :2],
                               posture_qpos=shifted_seed, previous_qpos=shifted_previous, max_step_rad=.12)
        joint_difference = float(np.max(np.abs(shifted.qpos[20:] - reference.qpos[20:])))
        error_difference = float(np.max(np.abs(shifted.position_errors_m - reference.position_errors_m)))
        assert joint_difference < 1e-8 and error_difference < 1e-8
        checks.append(dict(translation_m=translation.tolist(), arm_joint_max_abs_difference_rad=joint_difference,
                           ik_error_max_abs_difference_m=error_difference))
    ik_steps = np.abs(np.diff(trace['ik_target'], axis=0))
    pd_steps = np.abs(np.diff(trace['target'][:, 13:], axis=0))
    limits = model.jnt_range[14:]
    report = json.loads((run / 'report.json').read_text())
    return dict(solver_path=str(ROOT / 'gear_sonic/utils/g1_true23_intent_arm_ik.py'),
                solver_sha256=sha(ROOT / 'gear_sonic/utils/g1_true23_intent_arm_ik.py'),
                model_path=str(model_path), model_sha256=sha(model_path), original_path=str(original_path),
                original_sha256=sha(original_path), source_run=str(run), trace_sha256=sha(run / 'trace.npz'),
                report_sha256=sha(run / 'report.json'), translation_tests=checks,
                tested_source_frame=frame, pose_only=True, physical_tracking_qualified=False,
                translation_interpretation='Absolute translation cancels between measured-body FK and original-root-relative hand goals. Set estimator root position to zero for this IK component; root orientation and joint angles are still required. Outer BFM position feedback is a separate GT/odometry dependency.',
                feedforward=float(report['arm_ik_velocity_feedforward']),
                adjacent_ik_pose_step_max_rad=float(ik_steps.max()),
                adjacent_final_pd_target_step_max_rad=float(pd_steps.max()),
                final_pd_target_steps_over_point12_count=int(np.sum(pd_steps > .12 + 1e-9)),
                final_pd_target_native_bound_distance_min_rad=float(np.minimum(trace['target'][:, 13:] - limits[:, 0], limits[:, 1] - trace['target'][:, 13:]).min()),
                bound_interpretation='The .12 rad bound and optional margin apply to solved IK position. Velocity feedforward is added afterward; final PD targets are clipped only to native joint range. No physical velocity bound is inferred from target steps.')


result = dict(completion=completion_witness(), measured_arm=arm_witness(),
              hardware_authorized=False, source_files_modified=False)
(BASE / 'replay_completion_and_measured_arm_witness.json').write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps(result, indent=2, allow_nan=False))
