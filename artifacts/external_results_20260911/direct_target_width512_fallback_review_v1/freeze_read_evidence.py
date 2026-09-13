"""Record bounded read-only source/report inventory; no task-array libraries."""
import hashlib
import json
from pathlib import Path

NEW = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
MPC = Path('E:/codex_sonic_runtime/mpc_student_20260910')
BASE = NEW / 'direct_target_width512_fallback_review_v1'
relative = [
    'fast_controller_evidence_review_v1/README.md',
    'feedback_gain_supervision_review_v1/README.md',
    'three_expert_feedback_audit_v1/README.md',
    'fresh_student_expert_adapter_plan_v1/README.md',
    'fresh_expert_labels_resume_v1/README.md',
    'student_actual_oracle_control1_resume1001_v1/README.md',
    'bfm_entry250_actual_oracle_v1/README.md',
    'one_step_policy_branch_collection_resume2969_v1/RESULTS.md',
    'one_step_policy_branch_collection_v1/request.json',
    'direct_target_full_state_secants_v1/DESIGN.md',
    'direct_target_response_saved_semantics_review_v1/results_v1/report.json',
]
paths = [NEW / x for x in relative]
for name in ('robust_feedback_r004_barrier_v1', 'robust_feedback_r004_no_barrier_nominal_v1', 'robust_feedback_r004_no_barrier_perturbed_v1'):
    paths += [MPC / name / file for file in ('report.json', 'request.json', 'feedback_snapshot.py', 'evaluator_snapshot.py')]
paths += [MPC / 'student_3s_fit_v1' / x for x in ('student_snapshot.py', 'metrics.json', 'replay_nominal/report.json')]
paths += [MPC / 'teacher_native323_full_perturbed_v1/report.json']
paths += [BASE / 'README.md', Path(__file__)]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
pins = {p.as_posix(): sha(p) for p in paths}
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
full = read(MPC / 'teacher_native323_full_perturbed_v1/report.json')
assert full['eligible_cases'] == 0 and full['total_cases'] == 17
trials = {}
for name, count in [('robust_feedback_r004_barrier_v1', 3), ('robust_feedback_r004_no_barrier_nominal_v1', 1), ('robust_feedback_r004_no_barrier_perturbed_v1', 2)]:
    report = read(MPC / name / 'report.json')
    assert report['strict_successes'] == 0 and len(report['cases']) == count
    trials[name] = dict(strict_successes=0, cases=count, feedback_p95_ms=[r['feedback_p95_ms'] for r in report['cases']])
assert all(sha(Path(p)) == digest for p, digest in pins.items())
report = dict(kind='bounded_source_and_saved_JSON_fallback_evidence_review', evidence_review_completed=True,
    input_sha256=pins, fixed_sequence_full_walk002_eligible=[0,17], prior_capped_feedback=trials,
    genuine_failed_student_replan='final20000 residual student precontrol1; qualified original lifecycle plus hold',
    query250_origin='actual original BFM250 prefix, not later learned departure',
    physical_labels='3054 independent native one-step branches; fixed committed-map labels; zero fresh MPC queries',
    full58_labels='354612 independent signed one-axis fixed-map probes at 3057 nominal centers',
    recent_failed_direct_context_fresh_replans_found=False,
    search_scope='Inspected listed source/report paths and prior qualified evidence; not an exhaustive filesystem nonexistence proof.',
    recommendation='One prespecified early actual width departure fresh-expert feasibility/continuation query, after saved semantics and separate selection; then consider qualified DAgger data.',
    goal_conditioned_local_map_selector='Not found among inspected executed experiments; source-only hypothesis with legitimate state/context/preview inputs, no reference-index selection.',
    task_array_loads=0, checkpoint_loads=0, model_calls=0, native_steps=0, optimizer_updates=0, replans=0,
    new_execution_selected=False)
with (BASE / 'report.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(dict(report_sha256=sha(BASE/'report.json'), subjects=len(pins))))
