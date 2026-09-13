"""Twelve actual states against matching-clock saved committed maps; pure arrays."""
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
def paths():return dict(actual=NEW/'velocity_chord_student_evaluation_v1/nominal/trace.npz',
    centers=NEW/'velocity_chord_student_v1/generation/centers.npz',
    generation_receipt=NEW/'velocity_chord_student_v1/generation_frozen_inputs_v2.json',
    generation_audit=NEW/'velocity_chords_independent_v1/report.json',
    core=NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py',
    prior_outcome_audit=NEW/'student_velocity_chord_saved_outcome_v1/report.json',
    contract=local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))
def freeze():
    assert not (BASE/'request.json').exists()
    p=paths();assert read(p['prior_outcome_audit'])['passed']
    assert sha(p['actual'])=='1e8e44a6c405a86138558ea89681fb225b73b2be969e2e7a608903ebb5f80872'
    write(BASE/'request.json',dict(source_sha256=sha(__file__),input_sha256={str(v):sha(v) for v in p.values()},
        paths={k:str(v) for k,v in p.items()},controls=list(range(250,262)),
        nominal_map_checks=12,actual_state_map_evaluations=12,model_inference_calls=0,physics_steps=0,optimizer_updates=0,new_queries=0,
        map_selection='Matching global control from qualified query250 branch: dataset2 centers2038..2049; planned state/target/gain and both clips fixed.'))
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
            actor_history_deviation_rms=rms(a['history'][control]-centers['history'][idx]),
            worst_target_error_joint=names[worst],worst_target_error_signed_rad=float(error[worst]),
            left_knee=dict(head_target=float(a['target'][control,3]),fixed_map_target=float(target[3]),same_clock_nominal_target=float(nominal_target[3]),feedback_raw=float(raw[3]),feedback=float(feedback[3])))
        records.append(row)
        for key,value in dict(control=control,map_target=target,feedback_raw=raw,feedback_clipped=fc,native_clipped=nc,actual_tangent=tangent,actual_minus_expert_tangent=deviation,head_minus_map_target=error).items():arrays[key].append(value)
    for p,h in request['input_sha256'].items():assert sha(local(p))==h
    np.savez_compressed(BASE/'arrays.npz',**arrays)
    report=dict(passed=True,request_sha256=sha(BASE/'request.json'),source_sha256=sha(__file__),rows=records,
        nominal_map_checks_exact=12,actual_state_map_evaluations=12,feedback_clip_rows=sum(bool(r['feedback_clipped_joints']) for r in records),
        native_clip_rows=sum(bool(r['native_clipped_joints']) for r in records),all_inputs_unchanged=True,arrays_sha256=sha(BASE/'arrays.npz'),
        limitation='These are unchanged matching-clock committed-plan feedback maps evaluated on recorded student states. The maps do not consume prior/history, replan, establish local feasibility, or provide expert truth for those off-trajectory states.',
        model_inference_calls=0,physics_steps=0,optimizer_updates=0,new_queries=0)
    write(BASE/'report.json',report);print(json.dumps(records[:4],indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    else:main()
