"""Read-only array assessment and additive outcome receipts; no inference/physics."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


binding_paths = ['frozen_inputs_v2.json', 'fit/zero_head.onnx',
                 'fit/zero_initialization.json', 'zero_parity/report.json',
                 'zero_parity/trace.npz']
binding = dict(kind='completed_zero_preflight_binding_v1',
               description='Additive hash binding of completed pre-optimizer verification; original files unchanged.',
               hashes={p: sha(BASE/p) for p in binding_paths},
               script_sha256=sha(__file__))
(BASE/'zero_parity/completed_binding.json').write_text(json.dumps(binding, indent=2)+'\n')

a = np.load(BASE/'nominal/trace.npz', allow_pickle=False)
c = read(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
report = read(BASE/'nominal/report.json')
lim = np.asarray(c['joint_limits'])
q = a['physics_qpos'][-1, 7:]
excess = np.maximum(lim[:, 0]-q, q-lim[:, 1])
j = int(np.argmax(excess))
assert np.isclose(excess[j], report['range_excess_max'], atol=0, rtol=0)
assert np.array_equal(a['physics_time'], a['physics_expected_time'])
assert not a['physics_warning_counts'].any()
assert a['physics_substeps'].tolist() == [10]*24
assert int(a['source_frame'][-1]) == 34
summary = dict(
    status='REJECTED: nominal native joint bound failure before source motion',
    fit_attempts=1, learned_rollout_attempts=1, DAgger_queries=0,
    source_controls_reached=0, lifecycle_complete=False,
    extension_attempted=False, hardware_qualified=False,
    failure_joint=c['joint_names'][j], joint_index=j,
    actual_joint_position_rad=float(q[j]), native_bounds_rad=lim[j].tolist(),
    applied_target_rad=float(a['target'][-1, j]),
    unclipped_base_target_rad=float(a['base_target'][-1, j]),
    learned_residual_rad=float(a['delta'][-1, j]),
    actual_range_excess_rad=float(excess[j]),
    complete_physical_control_intervals=24,
    failed_control_zero_based=23, failure_substep_one_based=10,
    physics_steps=240, simulated_seconds=float(a['physics_time'][-1]),
    failure_phase='initial_entry',
    clock_array_bit_exact=True, engine_warnings=[0]*8,
    history_preclip_action_abs_max=float(np.max(np.abs(a['action']))),
    history_preclip_action_components_outside_5=int(np.sum(np.abs(a['action'])>5)),
    learned_residual_abs_max_rad=float(np.max(np.abs(a['delta']))),
    all_control_targets_inside_native_bounds=bool(np.all((a['target']>=lim[:,0])&(a['target']<=lim[:,1]))),
    interpretation='Final target clipping does not guarantee actual joint limits. Output-range restriction was removed, but this single nominal fit still fails. Neither command capacity nor training error establishes a usable controller.',
    retained_state='nominal/trace.npz contains every precontrol full integration vector, history, previous action, and final failure integration/history.',
    hashes={p:sha(BASE/p) for p in ['frozen_inputs_v2.json','fit/report.json','fit/student_head.onnx','nominal/report.json','nominal/request.json','nominal/trace.npz']},
    document_script_sha256=sha(__file__),
)
(BASE/'outcome.json').write_text(json.dumps(summary,indent=2)+'\n')
(BASE/'README.md').write_text('''# One nominal fast-controller pilot — rejected

The range-complete BFM-conditioned student failed during initial entry: right knee crossed its native lower bound by 0.002097812845 rad after 24 controls / 240 physics steps / 0.48 seconds. No source motion reached. The 1569-control lifecycle and separate 250-control hold were not completed. No DAgger query, second fit, or additional rollout was launched.

The fixed 1069→256→256→23 linear head trained for 1000 steps (seed 773, CPU one thread) in 9.98 seconds. Applied-target training RMSE was 0.08973 rad in initial entry, 0.10213 in acquisition, 0.13823 during source, and 0.08279 during return. These are training-state metrics, not generalization or behavioral success. Per-joint errors remain in fit/report.json. ONNX versus Torch maximum output difference was 9.69e-7 rad.

Before fitting, the zero head produced exactly zero residual on all 1269 demonstration states and exactly matched the same frozen ONNX BFM prior for 100 physical controls / 1000 steps, including history and targets. zero_parity/completed_binding.json binds the completed check, graph, and frozen source receipt. This does not claim historical PyTorch-versus-ONNX trajectory equivalence.

The learned rollout used original native dynamics at 500 Hz, bounded native PD effort, and 50 Hz policy updates. Every actual substep was recorded. Warning counts and accumulated clock errors stayed zero; maximum speed ratio was 0.615725 and maximum effort ratio 1.0. Targets stayed within native bounds, but the right knee's actual state violated its lower bound. The failed 24th control completed all ten physical substeps; physical completion count does not mean it passed.

Policy timing during this short failed rollout was 9.02 / 12.96 / 50.37 ms (median / p95 / maximum), with one of 24 calls above 20 ms. This is not a qualified real-time path or a quiet-machine benchmark.

All precontrol integration states, BFM histories, previous actions, and final failure state remain in nominal/trace.npz. Qualified teacher labels used actual applied-MPC target normalization; student recurrent history used the disclosed combined preclip BFM-plus-residual action. Original full-body goal lookahead was retained: 0.74-second feature knots, up to 0.76-second raw reference support. No frame ID, time, or clip ID enters the network.

This single fit removed the old residual cap and used a qualified full walk003 demonstration, but still failed closed loop. Further oracle correction requires a separately reviewed actual-state query and a qualified full remaining continuation, return, and standing hold. A short feasible MPC horizon alone is not an expert-label qualification.
''', encoding='utf-8')
print(json.dumps(summary, indent=2))
