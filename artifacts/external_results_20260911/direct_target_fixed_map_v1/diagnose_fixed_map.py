"""All actual direct-target learned controls against matching-clock saved committed maps; pure arrays."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
def local(p):
    s=str(p).replace('\\','/')
    if sys.platform!='win32' and len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+s[2:]
    return Path(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))
def paths():return dict(actual=NEW/'direct_target_student_evaluation_v1/nominal/trace.npz',
    centers=NEW/'velocity_chord_student_v1/generation/centers.npz',
    generation_receipt=NEW/'velocity_chord_student_v1/generation_frozen_inputs_v2.json',
    generation_audit=NEW/'velocity_chords_independent_v1/report.json',
    core=NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py',
    prior_outcome_audit=NEW/'direct_target_saved_outcome_review_v1/report.json',
    root_physics=local(args.root_physics),
    actual_report=NEW/'direct_target_student_evaluation_v1/nominal/report.json',
    final_fit_report=NEW/'direct_target_student_v1/fit/report.json',
    contract=local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))
def freeze():
    assert not (BASE/'request.json').exists()
    p=paths();assert read(p['prior_outcome_audit'])['passed']
    assert args.actual_sha and args.root_physics_sha and args.outcome_sha
    assert sha(p['actual'])==args.actual_sha
    assert sha(p['root_physics'])==args.root_physics_sha
    assert sha(p['prior_outcome_audit'])==args.outcome_sha
    actual=load(p['actual']);controls=actual['global_control'][actual['controller_mode']==1].astype(int).tolist()
    assert controls and controls==list(range(250,250+len(controls))) and controls[-1]<1269
    assert actual['features'].shape==(len(actual['target']),1000)
    assert read(p['prior_outcome_audit'])['moving_controls']==len(controls)
    write(BASE/'request.json',dict(source_sha256=sha(__file__),input_sha256={str(v):sha(v) for v in p.values()},
        paths={k:str(v) for k,v in p.items()},controls=controls,
        nominal_map_checks=len(controls),actual_state_map_evaluations=len(controls),model_inference_calls=0,physics_steps=0,optimizer_updates=0,new_queries=0,
        map_selection='Matching global control from qualified query250 branch: dataset2 center2038 + control-250; planned state/target/gain and both clips fixed.'))
def main():
    request=read(BASE/'request.json');assert sha(__file__)==request['source_sha256']
    assert sys.platform!='win32' and np.__version__=='1.26.4'
    for p,h in request['input_sha256'].items():assert sha(local(p))==h
    assert not (BASE/'report.json').exists()
    paths={k:local(v) for k,v in request['paths'].items()}
    a=load(paths['actual']);centers=load(paths['centers']);contract=read(paths['contract']);names=contract['joint_names']
    tree=ast.parse(paths['core'].read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('quat_mul','quat_log')]
    planner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    body.append(next(n for n in planner.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
    scope={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(paths['core']),'exec'),scope)
    fake=SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
    def difference(x,y):return scope['difference'](fake,x[None],y[None])[0]
    limits=centers['joint_limits'];records=[];arrays={k:[] for k in ('control','map_target','feedback_raw','feedback_clipped','native_clipped','actual_tangent','actual_minus_expert_tangent','head_minus_map_target')}
    for control in request['controls']:
        idx=2038+control-250;assert centers['dataset'][idx]==2 and centers['control'][idx]==control
        plan=centers['planned_state'][idx];u=centers['planned_target'][idx];gain=centers['gain'][idx]
        nominal=np.r_[centers['qpos'][idx],centers['qvel'][idx]]
        nominal_raw=gain@difference(plan,nominal)
        nominal_target=np.clip(u+np.clip(nominal_raw,-.1,.1),limits[:,0],limits[:,1])
        exact(nominal_target,centers['expert_target'][idx])
        actual=np.r_[a['qpos'][control],a['qvel'][control]];tangent=difference(plan,actual)
        raw=gain@tangent;feedback=np.clip(raw,-.1,.1);preclip=u+feedback;target=np.clip(preclip,limits[:,0],limits[:,1])
        deviation=difference(nominal,actual);error=a['target'][control]-target
        fc=raw!=feedback;nc=preclip!=target;worst=int(np.abs(error).argmax())
        row=dict(control=control,plan_control=int(centers['plan_control'][idx]),plan_local=int(centers['plan_local'][idx]),
            head_vs_fixed_map_target_rmse_rad=rms(error),head_vs_same_clock_nominal_target_rmse_rad=rms(a['target'][control]-nominal_target),
            fixed_map_vs_nominal_target_change_rms_rad=rms(target-nominal_target),
            feedback_clipped_joints=[names[j] for j in np.flatnonzero(fc)],native_clipped_joints=[names[j] for j in np.flatnonzero(nc)],
            feedback_raw_max_abs=float(np.abs(raw).max()),feedback_clipped_max_abs=float(np.abs(feedback).max()),
            root_position_deviation_m=float(np.linalg.norm(deviation[:3])),root_orientation_deviation_rad=float(np.linalg.norm(deviation[3:6])),
            joint_pose_deviation_rms_rad=rms(deviation[6:29]),joint_velocity_deviation_rms_radps=rms(deviation[35:58]),
            joint_velocity_native_cap_normalized_rms=rms(deviation[35:58]/np.asarray(contract['native_velocity'])),
            root_linear_velocity_deviation_mps=float(np.linalg.norm(deviation[29:32])),root_angular_velocity_deviation_radps=float(np.linalg.norm(deviation[32:35])),
            prior_action_deviation_rms=rms(a['previous_action'][control]-centers['previous_action'][idx]),
            terminal_BFM_history_deviation_rms=rms(a['history'][control]-centers['history'][idx]),
            worst_target_error_joint=names[worst],worst_target_error_signed_rad=float(error[worst]),
            left_knee=dict(head_target=float(a['target'][control,3]),fixed_map_target=float(target[3]),same_clock_nominal_target=float(nominal_target[3]),feedback_raw=float(raw[3]),feedback=float(feedback[3])),
            left_ankle_pitch=dict(head_target=float(a['target'][control,4]),fixed_map_target=float(target[4]),same_clock_nominal_target=float(nominal_target[4]),feedback_raw=float(raw[4]),feedback=float(feedback[4]),actual_precontrol_q=float(a['qpos'][control,11]),actual_precontrol_v=float(a['qvel'][control,10]),nominal_q=float(centers['qpos'][idx,11]),nominal_v=float(centers['qvel'][idx,10])),
            left_ankle_roll=dict(head_target=float(a['target'][control,5]),fixed_map_target=float(target[5]),same_clock_nominal_target=float(nominal_target[5]),feedback_raw=float(raw[5]),feedback=float(feedback[5]),actual_precontrol_q=float(a['qpos'][control,12]),actual_precontrol_v=float(a['qvel'][control,11]),nominal_q=float(centers['qpos'][idx,12]),nominal_v=float(centers['qvel'][idx,11])))
        records.append(row)
        for key,value in dict(control=control,map_target=target,feedback_raw=raw,feedback_clipped=fc,native_clipped=nc,actual_tangent=tangent,actual_minus_expert_tangent=deviation,head_minus_map_target=error).items():arrays[key].append(value)
    for p,h in request['input_sha256'].items():assert sha(local(p))==h
    np.savez_compressed(BASE/'arrays.npz',**arrays)
    report=dict(passed=True,request_sha256=sha(BASE/'request.json'),source_sha256=sha(__file__),rows=records,
        nominal_map_checks_exact=len(records),actual_state_map_evaluations=len(records),feedback_clip_rows=sum(bool(r['feedback_clipped_joints']) for r in records),
        native_clip_rows=sum(bool(r['native_clipped_joints']) for r in records),all_inputs_unchanged=True,arrays_sha256=sha(BASE/'arrays.npz'),
        limitation='These are unchanged matching-clock committed-plan feedback maps evaluated on recorded student states. The maps do not consume prior/history, replan, establish local feasibility, or provide expert truth for those off-trajectory states. The direct learned head uses only1000 measured-state/prepared-goal features; reported prior/history differences are terminal-handoff bookkeeping, not head inputs.',
        model_inference_calls=0,physics_steps=0,optimizer_updates=0,new_queries=0)
    assert sha(__file__)==request['source_sha256']
    for p,h in request['input_sha256'].items():assert sha(local(p))==h
    write(BASE/'report.json',report);print(json.dumps(records[:4],indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--root-physics');parser.add_argument('--root-physics-sha');parser.add_argument('--actual-sha');parser.add_argument('--outcome-sha');args=parser.parse_args()
    if args.freeze:freeze()
    else:
        try:main()
        except BaseException as error:
            write(BASE/'failure.json',dict(error=repr(error),model_inference_calls=0,physics_steps=0,optimizer_updates=0,new_queries=0));raise
