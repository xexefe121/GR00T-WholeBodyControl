"""Saved FP64 states versus immutable same-clock query250 maps; no engine/model imports."""
import argparse, ast, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
GROUPS=(('root_position',0,3),('root_rotation',3,6),('joint_position',6,29),
        ('root_linear_velocity',29,32),('root_angular_velocity',32,35),('joint_velocity',35,58))

def local(value):
    value=str(value).replace('\\','/')
    if sys.platform!='win32' and len(value)>2 and value[1]==':':value='/mnt/'+value[0].lower()+value[2:]
    return Path(value)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def rms(value):return float(np.sqrt(np.mean(np.asarray(value,np.float64)**2)))
def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def difference_function(core):
    tree=ast.parse(core.read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('quat_mul','quat_log')]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    body.append(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(core),'exec'),namespace)
    holder=SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
    return lambda x,y:namespace['difference'](holder,x[None],y[None])[0]

def committed_target(difference,plan_state,plan_target,gain,qpos,qvel,limits):
    # Preserve the reviewed saved_committed_map full product and both nested clips.
    actual=np.r_[qpos,qvel]
    tangent=difference(plan_state,actual)
    raw=gain@tangent
    correction=np.clip(raw,-.1,.1)
    preclip=plan_target+correction
    target=np.clip(preclip,limits[:,0],limits[:,1])
    return target,raw,correction,preclip

def grouped(gain,tangent):
    assert gain.shape==(23,58) and tangent.shape==(58,)
    parts=np.stack([gain[:,start:end]@tangent[start:end] for _,start,end in GROUPS])
    full=gain@tangent
    error=float(np.max(np.abs(parts.sum(axis=0)-full)))
    # Grouped products change accumulation order; they never produce the command.
    scale=max(1.,float(np.max(np.sum(np.abs(gain*tangent[None,:]),axis=1))))
    assert error<=64*np.finfo(np.float64).eps*scale
    other=gain[:,:35]@tangent[:35]
    velocity=gain[:,35:]@tangent[35:]
    return parts,other,velocity,error

def paths():
    return dict(actual=NEW/'direct_target_fp64_export_evaluation_v2/nominal/trace.npz',
        actual_report=NEW/'direct_target_fp64_export_evaluation_v2/nominal/report.json',
        semantic_report=NEW/'direct_target_fp64_saved_semantics_review_v1/report.json',
        semantic_rows=NEW/'direct_target_fp64_saved_semantics_review_v1/actual_rows.npz',
        root_physics=NEW/'direct_target_fp64_independent_physics_v1/report.json',
        centers=NEW/'velocity_chord_student_v1/generation/centers.npz',
        generation_receipt=NEW/'velocity_chord_student_v1/generation_frozen_inputs_v2.json',
        generation_audit=NEW/'velocity_chords_independent_v1/report.json',
        core=NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py',
        committed_map_source=NEW/'velocity_chord_student_v1/source_snapshot_v2/saved_committed_map.py',
        prior_map_source=NEW/'direct_target_fixed_map_v1/diagnose_fixed_map.py',
        export_report=NEW/'direct_target_fp64_export_v1/export/report.json',
        export_review=NEW/'direct_target_fp64_export_final_review_v1/review.json',
        contract=local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))

def freeze():
    assert not (BASE/'request.json').exists() and not (BASE/'report.json').exists()
    p=paths()
    known={'actual':'c5829d6e54bc5791bf9e9409d055496f7e94402c9af8e39ec83cf3b654ac4cae',
        'semantic_report':'64f0b4382d21109420bfbcc2dace856655ac8e4a9d0fbff1d48b39792a784506',
        'root_physics':'188e3521cf110ef0a27ad1ed77196aa2587082bf078b0bf318a23edb453d36e4',
        'centers':'5b07595d07e262f2ad236565ec62483d600843fe27a9f4a781974dac4b0608c7',
        'generation_audit':'2cc2346257ccedbc09ac406f5e4fcd8dc1b68851198c76646de28aebc4d0fc70'}
    for key,digest in known.items():assert sha(p[key])==digest
    s=read(p['semantic_report']);assert s['passed'] and s['moving_controls']==42
    assert s['actual_rows_sha256']==sha(p['semantic_rows']) and s['trace_sha256']==known['actual']
    original=ast.parse(p['committed_map_source'].read_text())
    mine=ast.parse(Path(__file__).read_text())
    get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='committed_target')
    assert ast.dump(get(original),include_attributes=False)==ast.dump(get(mine),include_attributes=False)
    write(BASE/'request.json',dict(source_sha256=sha(__file__),test_sha256=sha(BASE/'test_groups.py'),
        input_sha256={str(v):sha(v) for v in p.values()},paths={k:str(v) for k,v in p.items()},
        controls=list(range(250,292)),groups=GROUPS,nominal_map_checks=42,actual_state_map_evaluations=42,
        group_decomposition='Both plan-relative actual tangent and actual-minus-nominal tangents in the SAME plan chart. Full gain product commands retained; grouped sum checked with a roundoff bound only.',
        model_inference_calls=0,BFM_calls=0,physics_steps=0,optimizer_updates=0,replans=0))

def main():
    assert sys.platform!='win32' and np.__version__=='1.26.4'
    request_path=BASE/'request.json';request_sha=sha(request_path);request=read(request_path)
    assert sha(__file__)==request['source_sha256'] and sha(BASE/'test_groups.py')==request['test_sha256']
    assert not (BASE/'report.json').exists() and not (BASE/'arrays.npz').exists()
    for path,digest in request['input_sha256'].items():assert sha(local(path))==digest
    p={k:local(v) for k,v in request['paths'].items()}
    a=load(p['actual']);s=load(p['semantic_rows']);c=load(p['centers']);contract=read(p['contract'])
    difference=difference_function(p['core']);limits=c['joint_limits'];caps=np.asarray(contract['native_velocity'])
    assert caps.shape==(23,) and np.all(caps>0)
    exact(s['control'],np.arange(250,292,dtype=s['control'].dtype))
    assert a['features'].shape==(292,1000)
    arrays={};records=[]
    for j,control in enumerate(request['controls']):
        idx=2038+control-250
        assert c['dataset'][idx]==2 and c['control'][idx]==control
        assert int(c['plan_control'][idx])+int(c['plan_local'][idx])==control
        for key in ('qpos','qvel','target','features','previous_action','history'):
            if key in s:exact(a[key][control],s[key][j])
        plan=c['planned_state'][idx];u=c['planned_target'][idx];gain=c['gain'][idx]
        nominal=np.r_[c['qpos'][idx],c['qvel'][idx]];actual=np.r_[a['qpos'][control],a['qvel'][control]]
        nominal_target,nominal_raw,_,_=committed_target(difference,plan,u,gain,c['qpos'][idx],c['qvel'][idx],limits)
        exact(nominal_target,c['expert_target'][idx]);exact(nominal_target,s['teacher_target'][j])
        target,raw,correction,preclip=committed_target(difference,plan,u,gain,a['qpos'][control],a['qvel'][control],limits)
        tangent=difference(plan,actual);nominal_tangent=difference(plan,nominal)
        change=tangent-nominal_tangent
        # difference(nominal,actual) is for physical scale only, NOT the rotational contribution delta.
        departure=difference(nominal,actual)
        abs_parts,abs_other,abs_v,abs_err=grouped(gain,tangent)
        delta_parts,delta_other,delta_v,delta_err=grouped(gain,change)
        raw_change=raw-nominal_raw
        scale=max(1.,float(np.max(np.abs(raw))),float(np.max(np.abs(nominal_raw))),float(np.max(np.abs(gain@change))))
        assert np.max(np.abs(gain@change-raw_change))<=128*np.finfo(np.float64).eps*scale
        value=dict(control=control,center_index=idx,plan_control=int(c['plan_control'][idx]),plan_local=int(c['plan_local'][idx]),
            head_vs_fixed_map_rmse_rad=rms(a['target'][control]-target),head_vs_nominal_rmse_rad=rms(a['target'][control]-nominal_target),
            map_vs_nominal_rmse_rad=rms(target-nominal_target),head_clipped=bool(np.any(s['raw_proposal'][j]!=a['target'][control])),
            fixed_feedback_clip_joints=[contract['joint_names'][k] for k in np.flatnonzero(raw!=correction)],
            fixed_native_clip_joints=[contract['joint_names'][k] for k in np.flatnonzero(preclip!=target)],
            joint_velocity_departure_rms_radps=rms(departure[35:]),
            joint_velocity_native_cap_normalized_rms=rms(departure[35:]/caps),
            joint_velocity_native_cap_normalized_max=float(np.max(np.abs(departure[35:]/caps))),
            joint_position_departure_rms_rad=rms(departure[6:29]),
            raw_actual_group_rms_rad={name:rms(abs_parts[k]) for k,(name,_,_) in enumerate(GROUPS)},
            raw_change_group_rms_rad={name:rms(delta_parts[k]) for k,(name,_,_) in enumerate(GROUPS)},
            raw_actual_other35_rms_rad=rms(abs_other),raw_actual_velocity23_rms_rad=rms(abs_v),
            raw_change_other35_rms_rad=rms(delta_other),raw_change_velocity23_rms_rad=rms(delta_v),
            raw_change_other35_larger=bool(rms(delta_other)>rms(delta_v)),raw_change_full_rms_rad=rms(raw_change),
            grouped_accumulation_max_error_rad=max(abs_err,delta_err))
        records.append(value)
        values=dict(control=control,center_index=idx,map_target=target,nominal_target=nominal_target,head_target=a['target'][control],
            gain=gain,planned_state=plan,planned_target=u,actual_tangent=tangent,nominal_tangent=nominal_tangent,
            plan_chart_tangent_change=change,physical_departure=departure,feedback_raw=raw,nominal_feedback_raw=nominal_raw,
            feedback_correction=correction,preclip=preclip,feedback_raw_change=raw_change,
            raw_actual_groups=abs_parts,raw_change_groups=delta_parts,raw_change_other35=delta_other,raw_change_velocity23=delta_v)
        for key,val in values.items():arrays.setdefault(key,[]).append(val)
    for path,digest in request['input_sha256'].items():assert sha(local(path))==digest
    assert sha(request_path)==request_sha and sha(__file__)==request['source_sha256']
    np.savez_compressed(BASE/'arrays.npz',**arrays)
    subsets={'all42':records,'departed41':records[1:],'before_first_clip_251_255':records[1:6]}
    summary={name:dict(rows=len(rows),other35_larger_rows=sum(r['raw_change_other35_larger'] for r in rows),
        raw_change_other35_rms_rad=rms([r['raw_change_other35_rms_rad'] for r in rows]),
        raw_change_velocity23_rms_rad=rms([r['raw_change_velocity23_rms_rad'] for r in rows]),
        by_group_rms_rad={g:rms([r['raw_change_group_rms_rad'][g] for r in rows]) for g,_,_ in GROUPS}) for name,rows in subsets.items()}
    report=dict(passed=True,source_sha256=sha(__file__),request_sha256=request_sha,arrays_sha256=sha(BASE/'arrays.npz'),
        nominal_map_checks_exact=42,actual_state_map_evaluations=42,summary=summary,rows=records,
        feedback_clip_rows=sum(bool(r['fixed_feedback_clip_joints']) for r in records),native_clip_rows=sum(bool(r['fixed_native_clip_joints']) for r in records),
        all_inputs_unchanged=True,model_inference_calls=0,BFM_calls=0,physics_steps=0,optimizer_updates=0,replans=0,
        limitations=['Maps are stale same-clock committed query250 plans, not replanned expert truth or feasibility certificates.',
        'Group values decompose PRECLIP linear feedback. Their norms are not additive attribution percentages; cancellation and subsequent two nonlinear clips matter.',
        'Other35 includes root and joint position and root velocities. Explicit velocity probes cover only joint_velocity23; physical one-step labels already contain some coupled departures.',
        'Direct1000 features omit prior/history; those remain terminal-handoff bookkeeping. No models were called for this audit.'])
    write(BASE/'report.json',report)
    print(json.dumps(dict(report_sha256=sha(BASE/'report.json'),summary=summary,first_rows=records[:3]),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    else:
        try:main()
        except BaseException as error:
            write(BASE/'failure.json',dict(error=repr(error),model_inference_calls=0,physics_steps=0));raise
