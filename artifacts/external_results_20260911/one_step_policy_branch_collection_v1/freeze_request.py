"""Freeze selected collector source/inputs only. No graphs, model or dynamics."""
import ast,json,hashlib,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SRC=BASE/'source_snapshot_v1';MEMORY=NEW/'prior_memory_intervention_v1'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
assert not SRC.exists() and not (BASE/'request.json').exists()
checks=read(BASE/'pure_checks.json');assert checks['passed'] and checks['all_source_rows_verified']==3054 and checks['tests']==9
for key,digest in checks['source_sha256'].items():assert sha(BASE/'draft'/key)==digest
for p in (BASE/'draft').rglob('*.py'):
    ast.parse(p.read_text());dest=SRC/p.relative_to(BASE/'draft');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
old=read(MEMORY/'request.json');inputs=dict(old['input_sha256']);paths=dict(old['paths'])
for key in ('fixed_map_report','fixed_map_arrays'):paths.pop(key)
bundle=Path(paths['contract']).parent
paths.update(bundle=bundle.as_posix(),trace0=(NEW.parent/'sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/trace.npz').as_posix(),
 trace1=(NEW/'student_actual_oracle_control1_resume1001_v1/nominal/trace.npz').as_posix(),trace2=(NEW/'bfm_entry250_actual_oracle_v1/nominal/trace.npz').as_posix(),
 old_snapshots=(NEW/'old_expert_prefix_snapshots_v2/capture/control_snapshots.npz').as_posix(),old_report=(NEW/'old_expert_prefix_snapshots_v2/capture/report.json').as_posix(),
 witness=(NEW/'velocity_chord_student_evaluation_v1/head_witness/witness.npz').as_posix())
assert sha(paths['old_snapshots'])=='84aaf88c70d6559827839810424761ae923ca0ddce10cf54e8b3c7594c518e29'
assert sha(paths['old_report'])=='fb0226872958f414b9fdf263c54ca340733a697b7fea82c86215f872cbd28e30'
assert sha(SRC/'native_forecast.py')=='81af01274f55c703850cb3ec3801bb41912e3f3786440af5099b769b7bd28824'
assert sha(SRC/'stateless_adapter.py')=='5f932ab291ebed5abbe900f793869420208f51f688fe64c91e21fe712d964848'
for key,path in paths.items():
    if key not in ('bundle','onnx_dependencies'):inputs[path]=sha(path)
old_capture=NEW/'old_expert_prefix_snapshots_v2'
for path,digest in read(old_capture/'capture_request.json')['input_hashes'].items():inputs[path]=digest
extra=[old_capture/'completion_verification.json',old_capture/'capture_request.json',old_capture/'capture_process/launch_receipt.json',
 NEW/'one_step_distillation_design_review_v1/review.json',NEW/'one_step_collector_draft_review_v1/review.json',
 Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/PROPOSED_ONE_STEP_DISTILLATION.md'),
 NEW/'previous_command_perturbation_feasibility_v1/behavior_report.json',
 NEW/'prior_memory_intervention_v1/results/report.json',NEW/'prior_memory_intervention_v1/request.json',
 NEW/'phase_student_preallocated_forecast_v2/results/report.json',
 BASE/'pure_checks.json',BASE/'pure_checks.log',BASE/'check_saved_inputs.py',BASE/'run_durable.ps1',Path(__file__)]
for path in extra:inputs[path.as_posix()]=sha(path)
for path,digest in inputs.items():assert sha(path)==digest,path
sources={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*.py')}
write(BASE/'request.json',dict(kind='selected_one_control_physical_policy_branch_collection',source_directory=SRC.as_posix(),source_sha256=sources,input_sha256=inputs,paths=paths,
 rows=3054,row_order='dataset0/1/2 each startcontrol250..1267; center=dataset*1019+control-250; successorcenter=center+1; successorcontrol=c+1; goalframe=c+12',
 policy_order='calibration row2036 first, then natural rows excluding2036; no repeat',
 native_step_ceiling=61080,graph_call_ceiling=dict(backward=3054,actor=3054,head=6108),total_graph_call_ceiling=12216,
 nominal_first='All3054 original10-step transitions/fullsuccessor291 byteexact before any graph call or policybranch.',
 calibration='Exact query250 input and one WSL batch1 output witness; actual250 command/history and all10native samples/full251291/base/state/features/head before credit.',
 native='Unchanged NativeForecast81af0127, 500Hz manualnativePD, strict every2ms, onecontrol10steps maximum; separate actual nativeattempted/returned/captured counters.',
 history='Original measured state/incoming rawprior pushes exactly once newest-first. Outgoing rawcombined command is separate current prior at endpoint. Applied-normalized action is diagnostic only.',
 label='Same-clock saved committed plan original signed full K@difference, feedbackclip+/-0.1 andnative targetclip; residual relative to newly recomputed unclipped BFMbase. No replanning or feasibility claim.',
 failure='Strict-failed policybranch retained, labelinvalid and no endpointgraphs; nativeexception stops entire collection preserving partial states/counters; no retries.',
 output='collection/report.json plus collection/data/manifest.json maps logical array keys to relative path/shape/dtype/SHA; native_rows.jsonl andgraph_calls.jsonl independently bound.',
 optimizer_updates=0,new_mpc_queries=0,new_canonical_trials=0,head_batch=1,onnxruntime='1.23.2',numpy='1.26.4',mujoco='3.2.3',threads=1))
write(BASE/'clearance_DRAFT.json',dict(approved=False,request_sha256=sha(BASE/'request.json'),launcher_sha256=sha(BASE/'run_durable.ps1'),
 rows=3054,native_step_ceiling=61080,total_graph_call_ceiling=12216,review_path=None,review_sha256=None))
print(json.dumps(dict(request_sha256=sha(BASE/'request.json'),sources=len(sources),inputs=len(inputs),launcher_sha256=sha(BASE/'run_durable.ps1'))))
