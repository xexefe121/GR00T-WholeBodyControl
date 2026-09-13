"""Saved-array/AST arithmetic only: no production module imports or dynamics."""
from pathlib import Path
from types import SimpleNamespace
import ast, hashlib, json
import numpy as np

OUT=Path(__file__).resolve().parent
NEW=OUT.parent
EXPERT=NEW/'student_actual_oracle_control1_resume1001_v1'
SRC=NEW/'student_actual_oracle_control1_v1/source_snapshot_v2'
CORE=SRC/'gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py'
ARRAYS=Path(r'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\teleop_six_hour_20260910\mjbatch_native23_inputs_v1\prepared_model_arrays.npz')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'metrics.json').exists()
tree=ast.parse(CORE.read_text())
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('quat_mul','quat_log')]
planner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
selected.append(next(n for n in planner.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
scope={'np':np}
exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(CORE),'exec'),scope)
fake=SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
trace=np.load(EXPERT/'nominal/trace.npz',allow_pickle=False)
limits=np.load(ARRAYS,allow_pickle=False)['actuator_ctrlrange']
plans=json.loads((EXPERT/'nominal/plans.json').read_text())
rows=[]; files=[CORE,SRC/'run_actual_student_oracle.py',EXPERT/'nominal/trace.npz',EXPERT/'nominal/plans.json',ARRAYS,Path(__file__),NEW/'expert_target_saturation_diagnostic_v1/report.json',NEW/'fast_controller_evidence_review_v1/README.md',NEW/'expert_resumed_root_qualification_v1/qualification.json',NEW/'fast_controller_aggregate_fit_v1/source_snapshot_v2/student_linear_runtime.py']
no_update=[]
for plan in plans:
 start=int(plan['control']); count=int(plan['executed_controls'])
 path=EXPERT/'nominal/plans'/('plan_%05d.npz'%start); files.append(path)
 p=np.load(path,allow_pickle=False)
 if not any(x['accepted'] for x in plan['solver_feasibility']['iterations']): no_update.append(start)
 for local in range(count):
  control=start+local; actual=np.r_[trace['qpos'][control],trace['qvel'][control]]
  delta=scope['difference'](fake,p['nominal_states'][local:local+1],actual[None])[0]
  gain=p['gains'][local]; raw=gain@delta; correction=np.clip(raw,-.1,.1)
  preclip=p['targets'][local]+correction; target=np.clip(preclip,limits[:,0],limits[:,1])
  assert np.array_equal(target,trace['target'][control]),control
  rows.append(dict(control=control,plan_control=start,local=local,target_reconstruction_error=float(np.max(np.abs(target-trace['target'][control]))),max_abs_K=float(np.max(np.abs(gain))),jointpos_K_max=float(np.max(np.abs(gain[:,6:29]))),jointvel_K_max=float(np.max(np.abs(gain[:,35:58]))),actual_tangent_max=float(np.max(np.abs(delta))),raw_feedback_max=float(np.max(np.abs(raw))),feedback_clipped_components=int(np.count_nonzero(raw!=correction)),native_target_clipped_components=int(np.count_nonzero(target!=preclip)),gain_is_zero=bool(not np.any(gain))))
assert [r['control'] for r in rows]==list(range(1,1269))
metrics=dict(kind='saved_expert_feedback_gain_arithmetic_only',plans=len(plans),used_controls=len(rows),plans_without_accepted_update=len(no_update),no_update_plan_controls=no_update,zero_gain_controls=sum(r['gain_is_zero'] for r in rows),max_target_reconstruction_error=max(r['target_reconstruction_error'] for r in rows),raw_feedback_max=max(r['raw_feedback_max'] for r in rows),feedback_clipped_components=sum(r['feedback_clipped_components'] for r in rows),feedback_clipped_controls=sum(r['feedback_clipped_components']>0 for r in rows),native_target_clipped_components=sum(r['native_target_clipped_components'] for r in rows),max_abs_K=max(r['max_abs_K'] for r in rows),p95_control_max_abs_K=float(np.percentile([r['max_abs_K'] for r in rows],95)),jointpos_K_max=max(r['jointpos_K_max'] for r in rows),jointvel_K_max=max(r['jointvel_K_max'] for r in rows),controls_with_some_1mrad_jointpos_direction_correction_over_01=sum(r['jointpos_K_max']*.001>.1 for r in rows),controls_with_some_001rads_jointvel_direction_correction_over_01=sum(r['jointvel_K_max']*.01>.1 for r in rows),actual_tangent_max=max(r['actual_tangent_max'] for r in rows),physical_perturbations_tested=0,reviewer_inference_calls=0,reviewer_optimizer_calls=0,reviewer_physics_steps=0)
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
(OUT/'per_control.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'input_hashes.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2)+'\n')
print(json.dumps(metrics,indent=2))
