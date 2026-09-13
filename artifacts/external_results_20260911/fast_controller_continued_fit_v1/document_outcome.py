"""Document the single continued fit and its single failed physical assessment."""
import hashlib,json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
ORIGINAL=BASE.parent/'fast_controller_nominal_pilot_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
c=read('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
a=np.load(BASE/'nominal/trace.npz',allow_pickle=False)
l=np.load(BASE/'labels/labels.npz',allow_pickle=False)
f=np.load(BASE/'fit/teacher_fit.npz',allow_pickle=False)
v=a['physics_qvel'][-1,6:];j=int(np.argmax(abs(v)/c['native_velocity']))
fit=read(BASE/'fit/report.json');nominal=read(BASE/'nominal/report.json')
assert a['physics_substeps'].tolist()==[10]*15+[6]
assert np.array_equal(a['physics_time'],a['physics_expected_time'])
assert not a['physics_warning_counts'].any()
initial={k:bool(np.array_equal(a[k][0],l[k][0])) for k in ['features','state','history','previous_action','base_target']}
assert all(initial.values())
same_time=[]
for i in [0,1,5,10,15]:
    same_time.append(dict(control=i,precontrol_time_seconds=.02*i,
        qpos_max_delta_to_teacher=float(np.max(abs(a['qpos'][i]-l['teacher_qpos'][i]))),
        qvel_max_delta_to_teacher=float(np.max(abs(a['qvel'][i]-l['teacher_qvel'][i]))),
        teacher_input_target_RMSE_rad=float(np.sqrt(np.mean(f['applied_error'][i]**2))),
        actual_rollout_target_RMSE_rad=float(np.sqrt(np.mean((a['target'][i]-l['expert_target'][i])**2)))))
result=dict(status='REJECTED before source: right hip roll speed exceeded native20rad/s',
    total_fit_steps=fit['steps'],fit_stop_reason=fit['stop_reason'],fit_stop_criteria_met=False,
    continuation_fit_attempts=1,ordinary_final_checkpoint_used=True,intermediate_physics_evaluations=0,
    learned_nominal_attempts=1,full_lifecycle_completed=False,source_controls_reached=0,extension_attempts=0,DAgger_queries=0,
    completed_physical_controls=15,partial_control_substeps=6,physics_steps=156,seconds=float(a['physics_time'][-1]),
    failure_joint=c['joint_names'][j],actual_joint_velocity_rad_s=float(v[j]),native_velocity_bound_rad_s=c['native_velocity'][j],
    velocity_ratio=float(abs(v[j])/c['native_velocity'][j]),
    final_applied_target_rad=float(a['target'][-1,j]),final_BFM_base_target_rad=float(a['base_target'][-1,j]),
    final_learned_delta_rad=float(a['delta'][-1,j]),
    range_excess_max=nominal['range_excess_max'],actual_effort_ratio_max=nominal['effort_ratio_max'],
    warnings=[0]*8,physics_clock_exact=True,identical_initial_input=initial,
    same_time_diagnostics=same_time,
    full1269_fit_RMSE_rad=fit['final_fit_diagnostic']['full1269_applied_target_RMSE_rad'],
    first24_worst_joint_target_error_p95_rad=fit['final_fit_diagnostic']['maximum_first24_perjoint_absolute_error_p95_rad'],
    comparison='Improved teacher-state and initial-state fit did not stabilize this controller; small first-step departure grew rapidly. No further fit or DAgger inference follows from this diagnosis.',
    preserved_states='Every actual precontrol integration vector/history/previous action and final partial-control failure state are retained in nominal/trace.npz.',
    hashes={p:sha(BASE/p) for p in ['frozen_inputs_v2.json','fit/reconstruction_parity.json','fit/full_fit_metrics.json','fit/report.json','fit/student_head.onnx','fit/student_head.pt','nominal/report.json','nominal/trace.npz']},
    document_script_sha256=sha(__file__))
(BASE/'outcome.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
(BASE/'README.md').write_text('''# One supervised-fit continuation — rejected

The ordinary final 20,000-step student failed its single native physical attempt at 0.312 seconds: right hip-roll velocity reached -21.25169 rad/s against the native 20 rad/s limit. It completed 15 control intervals plus six substeps of the next interval (156 physics steps). No source motion was reached; the 1569-control lifecycle and separate continuous 250-control hold were not completed. No DAgger query or extra fit was run.

All 15 previous v2 source files, including runtime and evaluator, were copied byte for byte into source_snapshot_v3. Only continue_linear_fit.py was added. Same 1269 labels, features, normalization, network, joint-span output, phase sampling, seed 773, AdamW settings, and CPU one-thread execution were retained. The original failed pilot remains untouched.

Before step 1001, the original first 1000 updates were regenerated. Every model tensor, normalization/span, all ten logged losses, and all saved teacher prediction arrays matched exactly. Reconstructed optimizer moments and Torch/NumPy random states were saved, then restored for continuation. The original optimizer was not saved, so this is deterministic reconstruction evidence, not a comparison against nonexistent original optimizer bytes. fit/reconstruction_parity.json contains the receipt.

The single continuation ran to the declared maximum of 20,000 total steps, taking 179.37 additional seconds. Full/per-joint teacher metrics were recorded every 1000 steps. The early-stop conditions—full1269 applied-target RMSE <=0.01 rad AND each first24 joint's p95 absolute target error <=0.03 rad—never passed. Final values were 0.02684 and 0.15703 rad respectively. No intermediate physical tests or best-checkpoint selection occurred; the ordinary final actor was used. ONNX/Torch output difference was at most 1.82e-6 rad.

| Measure | Original 1000 steps | Final 20,000 steps |
|---|---:|---:|
| Initial input target RMSE | 0.10864 rad | 0.02409 rad |
| Source teacher-input target RMSE | 0.13823 rad | 0.02744 rad |
| Initial right-knee target error | -0.21701 rad | +0.02614 rad |
| Initial right ankle-pitch target error | -0.15803 rad | +0.03590 rad |
| Failed physical time | 0.480 s | 0.312 s |

At control 0, all feature/state/history/previous-action/base-target arrays still match the teacher input exactly. After one 20 ms interval, actual q differs from the teacher by up to 0.00782 rad and qvel by 0.51274 rad/s. Actual-state target RMSE is then 0.09801 rad, compared with 0.02306 rad on the saved corresponding teacher input. By control 5, these target errors are 0.75063 and 0.02079 rad. Better fitting on the nominal trajectory did not prevent rapid error growth away from it.

At failure, the right hip-roll BFM base target was +0.14727 rad and the learned residual -2.29098 rad, producing the applied target -2.14370 rad. Actual speed, rather than commanded target range, triggered the stop. Actual range excess remained zero, actual effort ratio never exceeded 1.0, every engine warning counter was zero, and recorded time matched the independently accumulated 2 ms clock exactly.

The short failed rollout's policy median/p95/maximum timing was 9.46/21.58/51.55 ms, with one of 16 calls above 20 ms. This is not a qualified real-time or quiet-machine benchmark. Every precontrol full integration state, history, and previous action, plus the final partial-control failure state, remains in nominal/trace.npz. Physical continuation and source qualification remain unachieved.
''',encoding='utf-8')
print(json.dumps(result,indent=2))
