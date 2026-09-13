"""Saved feedback/history algebra only; no model or native dynamics."""
from pathlib import Path
import hashlib,json
import numpy as np
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');OWNER=BASE/'physical_student_clipped_feedback_v1';OUT=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
pins={};checks=0
def pin(p,d=None):
    actual=sha(p)
    if d is not None:assert actual==d
    pins[str(p)]=actual
def eq(a,b):
    global checks
    a,b=np.asarray(a),np.asarray(b);checks+=1
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),checks
tp=OWNER/'nominal/trace.npz';op=BASE/'one_step_physical_student_evaluation_v1/nominal/trace.npz'
pin(tp,'454b8c577ef64edc1128cd8e5e9d295c2936f2ba854ddd9aeba0286ff112eaa7');pin(op,'38428ae27055be7c2e771c5a23056c858f004ef38f260a0b45a57662d8d0aae3')
cp=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json');pin(cp,'d1641ce4f2e9016f5008e5d4be3cc24ded46d7f9daf9cfddaab84095f98eb99b')
pr=BASE/'student_clipped_feedback_independent_physics_v1/report.json';pin(pr,'fe83ffcd1730ae2ccc16b2e5e4286278bcfcdb6ff7067a880438154e340ec047')
for p in (OWNER/'completion_verification.json',OWNER/'nominal/report.json',OWNER/'evaluation_process/exit.json',BASE/'student_clipped_feedback_independent_intent_v1/report.json'):pin(p)
c=read(cp);t=load(tp);o=load(op);r=read(OWNER/'nominal/report.json');physical=read(pr)
assert physical['recorded_trace_reproduced_through_last_sample'] and physical['physics_steps']==3146
assert physical['feasible'] is False and r['completed_controls']==314 and r['attempted_controls']==315
assert read(OWNER/'evaluation_process/exit.json')['exit_code']==1
eq(t['global_control'],np.arange(315,dtype=np.int64));eq(t['physics_substeps'],np.r_[np.full(314,10,dtype=np.int64),6])
limits=np.asarray(c['joint_limits']);raw=t['base_target']+t['delta'];target=np.clip(raw,limits[:,0],limits[:,1]);eq(raw,t['raw_proposal']);eq(target,t['target'])
applied=((target-np.asarray(c['default_q']))*np.asarray(c['kp'])/(.25*np.asarray(c['training_effort']))).astype(np.float32)
native=raw!=target;mask=native.copy();mask[:250]=False
feedback=t['action'].copy();feedback[mask]=applied[mask]
for a,b in [(applied,t['actual_normalized_action']),(native,t['native_target_clip_mask']),(mask,t['feedback_clip_mask']),(feedback,t['feedback_action']),(t['action'],t['raw_combined_action']),(feedback[~mask],t['action'][~mask])]:eq(a,b)
history={k:np.zeros((4,n),np.float32) for k,n in dict(actions=23,base_ang_vel=3,dof_pos=23,dof_vel=23,projected_gravity=3).items()}
prior=np.zeros(23,np.float32)
for row in range(315):
    before=np.concatenate([history[k].reshape(-1) for k in sorted(history)])
    eq(before,t['control_history_before'][row]);eq(before,t['history'][row]);eq(prior,t['previous_action'][row]);eq(prior,t['control_previous_action_before'][row])
    state=t['state'][row]
    terms=dict(actions=prior,base_ang_vel=state[49:52],dof_pos=state[:23],dof_vel=state[23:46],projected_gravity=state[46:49])
    for k in history:
        history[k][1:]=history[k][:-1].copy();history[k][0]=terms[k]
    prior=feedback[row].copy()
for k in history:eq(history[k],t['final_history_'+k])
eq(prior,t['final_previous_action']);eq(t['final_recorded_controls'],np.asarray(315,dtype=np.int64))
for k in ('target','source_frame','global_control','controller_mode','state','history','previous_action','action','base_target','delta','features','physics_substeps','raw_proposal','actual_normalized_action','control_integration_before','control_history_before','control_previous_action_before'):eq(t[k][:265],o[k][:265])
for k in ('qpos','qvel'):eq(t[k][:266],o[k][:266])
for k in ('physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo'):eq(t[k][:2651],o[k][:2651])
for k in ('physics_torque','physics_actuator_torque','range_excess','velocity_ratio','effort_ratio','clock_error'):eq(t[k][:2650],o[k][:2650])
eq(t['control_integration_before'][265],o['control_integration_before'][265])
def firstdiff(a,b):return next((i for i,(x,y) in enumerate(zip(a,b)) if x.tobytes()!=y.tobytes()),None)
assert firstdiff(t['previous_action'],o['previous_action'])==265 and firstdiff(t['history'],o['history'])==266
pin(Path(__file__))
result=dict(saved_semantics_review_pass=True,lifecycle_pass=False,comparisons=checks,input_sha256=pins,original_prefix_native_samples=2650,
    original_precontrol265_full291_exact=True,all315_feedback_history_rows_exact=True,first_changed_prior=265,first_changed_actor_history=266,
    changed_feedback_commands=int(np.any(mask,axis=1).sum()),changed_feedback_components=int(mask.sum()),
    failure=r['failure'],failure_joint='left_knee_joint',source_controls=0,conditional_hold_executed=False,
    interpretation='The selected feedback semantics and exact pre-change prefix held. This single variant still failed before source motion; it does not establish a fix for early pre-clipping amplification.',
    reviewer_model_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0)
for p,d in pins.items():assert sha(p)==d
with (OUT/'report.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(pass_all=True,comparisons=checks,report_sha256=sha(OUT/'report.json'))))
