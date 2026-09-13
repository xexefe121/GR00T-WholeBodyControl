"""Saved JSON/source accounting only. No numerical task libraries."""
import hashlib
import json
from pathlib import Path
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=Path(__file__).resolve().parent
paths=[BASE/'BFM_FEASIBILITY.md',Path(__file__),
    NEW/'direct_target_student_v1/DESIGN.md',NEW/'direct_target_student_evaluation_v1/README.md',
    NEW/'fast_controller_evidence_review_v1/README.md',
    NEW/'physical_student_clipped_feedback_v1/RESULTS.md',
    NEW/'physical_student_clipped_feedback_v1/SELECTION.md',
    NEW/'one_step_physical_student_evaluation_v1/RESULTS.md',
    NEW/'one_step_physical_student_v1/source_snapshot_v1/student_linear_runtime.py',
    NEW/'direct_target_saved_diagnostics_v1/phase_timing.json']
trials={}
for name in ('one_step_physical_student_evaluation_v1','physical_student_clipped_feedback_v1','fast_controller_phase_fit_v1','velocity_chord_student_evaluation_v1'):
    path=NEW/name/'nominal/report.json';paths.append(path)
    r=json.loads(path.read_text(encoding='utf-8-sig'))
    assert r['policy_20ms_deadline_misses']==0 and r['failure'] is not None
    trials[name]={k:r[k] for k in ('policy_ms_p50_p95_max','policy_20ms_deadline_misses','attempted_controls','failure')}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={p.as_posix():sha(p) for p in paths}
report=dict(kind='BFM_architecture_and_recorded_proposal_budget_source_review',evidence_review_completed=True,
    zero_learned_BFM_scope='Selected direct-head runtime guard; no global original-user ban found in inspected artifacts.',
    original_user_wording_verified=False,
    prior_BFM_plus_head_trials=trials,
    timing_limit='Proposal only; no independent plant/IPC/full-cycle deadline qualification.',
    source_capacity_limit='Early 0.25*tanh cap rejected; later uncapped residual also failed behavior.',
    input_sha256=pins,task_array_loads=0,model_calls=0,native_steps=0,optimizer_updates=0,replans=0,
    current_runtime_contract_changed=False,new_execution_selected=False)
assert all(sha(Path(p))==h for p,h in pins.items())
with (BASE/'bfm_feasibility_report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(report_sha256=sha(BASE/'bfm_feasibility_report.json'),subjects=len(pins))))
