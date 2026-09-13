"""Saved plans and exact pure difference arithmetic; no labels/inference/physics."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import numpy as np

BASE=Path(__file__).resolve().parent;NEW=BASE.parent;OLD=NEW.parent/'sonic23_teleop_six_hour_20260910'
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
SOURCE=NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1'
CORE=SOURCE/'gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py'
PRODUCER=OLD/'mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def summary(x):
    x=np.asarray(x,np.float64)
    return dict(min=float(x.min()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))

def main():
    assert not (BASE/'report.json').exists()
    paths={};bind=lambda p:paths.update({str(p):sha(p)})
    for p in (CORE,Path(__file__),SOURCE/'run_actual_student_oracle.py',PRODUCER/'evaluate_g1_true23_mjbatch_mpc_snapshot.py',
        PRODUCER/'g1_true23_mjbatch_ilqr_core_snapshot.py',PRODUCER/'request.json',
        NEW/'feedback_gain_supervision_review_v1/README.md'):bind(p)
    core=ast.parse(CORE.read_text())
    pure=[n for n in core.body if isinstance(n,ast.FunctionDef) and n.name in ('quat_mul','quat_log')]
    klass=next(n for n in core.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    pure.append(next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
    oldcore=ast.parse((PRODUCER/'g1_true23_mjbatch_ilqr_core_snapshot.py').read_text())
    oldclass=next(n for n in oldcore.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    assert ast.dump(pure[-1])==ast.dump(next(n for n in oldclass.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
    scope={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=pure,type_ignores=[])),str(CORE),'exec'),scope)
    fake=SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
    contractpath=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json';bind(contractpath)
    c=read(contractpath);limits=np.asarray(c['joint_limits']);names=c['joint_names']
    configs=[('old',OLD/'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1',
        NEW/'fast_controller_nominal_pilot_v1/labels/labels.npz'),
        ('query1',NEW/'student_actual_oracle_control1_resume1001_v1',NEW/'fresh_expert_labels_resume_v1/labels/labels.npz'),
        ('query250',NEW/'bfm_entry250_actual_oracle_v1',NEW/'bfm_entry250_labels_v1/labels/labels.npz')]
    allrows=[];reports={}
    for name,expert,labelpath in configs:
        tracepath=expert/('trace.npz' if name=='old' else 'nominal/trace.npz');bind(tracepath);trace=load(tracepath)
        bind(labelpath);labels=load(labelpath);selected=np.flatnonzero((labels['control']>=250)&(labels['control']<1269))
        np.testing.assert_array_equal(labels['control'][selected],np.arange(250,1269))
        np.testing.assert_array_equal(labels['expert_target'][selected],trace['target'][250:1269])
        starts={};plans={}
        if name=='old':
            producerpath=PRODUCER/'trace.npz';bind(producerpath);producer=load(producerpath)
            for key in ('qpos','qvel'):np.testing.assert_array_equal(producer[key][:1270],trace[key][:1270])
            np.testing.assert_array_equal(producer['target'][:1269],trace['target'][:1269])
            planpath=PRODUCER/'plans.json';bind(planpath)
            for plan in read(planpath):
                for control in range(plan['control'],plan['control']+plan['controls_committed']):starts[control]=(plan['control'],plan)
        else:
            planpath=expert/'nominal/plans.json';bind(planpath)
            for plan in read(planpath):
                start=plan['control'];count=plan['executed_controls']
                if start+count<=250 or start>=1269:continue
                p=expert/'nominal/plans'/('plan_%05d.npz'%start);bind(p);plans[start]=load(p)
                for control in range(start,start+count):starts[control]=(start,plan)
        rows=[]
        for control in range(250,1269):
            start,plan=starts[control];local=control-start
            if name=='old':
                state=producer['planned_state'][control];target_plan=producer['planned_target'][control];gain=producer['feedback_gain'][control]
                accepted=None
            else:
                p=plans[start];state=p['nominal_states'][local];target_plan=p['targets'][local];gain=p['gains'][local]
                accepted=any(it['accepted'] for it in plan['solver_feasibility']['iterations'])
            actual=np.r_[trace['qpos'][control],trace['qvel'][control]]
            tangent=scope['difference'](fake,state[None],actual[None])[0]
            raw=gain@tangent;correction=np.clip(raw,-.1,.1);preclip=target_plan+correction
            target=np.clip(preclip,limits[:,0],limits[:,1]);np.testing.assert_array_equal(target,trace['target'][control])
            if name=='old':
                np.testing.assert_array_equal(raw,producer['feedback_correction_raw'][control])
                np.testing.assert_array_equal(correction,producer['feedback_correction_applied'][control])
            v=gain[:,35:58];q=gain[:,6:29]
            feedback_boundary=np.abs(np.abs(raw)-.1)<=1e-12
            target_boundary=np.minimum(np.abs(preclip-limits[:,0]),np.abs(preclip-limits[:,1]))<=1e-12
            moving=np.any(v!=0,axis=1)
            nondiff=(feedback_boundary|target_boundary)&moving
            smooth_active=(np.abs(raw)<.1)&(preclip>limits[:,0])&(preclip<limits[:,1])&~nondiff
            valid_zero=(~nondiff)&~smooth_active
            smooth_v=v*smooth_active[:,None]
            # A radius for staying within current interior feedback and target branches.
            margin=np.minimum(.1-np.abs(raw),np.minimum(preclip-limits[:,0],limits[:,1]-preclip))
            maxrow=np.max(np.abs(v),axis=1)
            bound_rows=(maxrow>0)&smooth_active
            radii=margin[bound_rows]/maxrow[bound_rows]
            axis_radius=float(radii.min()) if len(radii) and not nondiff.any() else None
            diag=np.diag(v)
            row=dict(dataset=name,control=control,plan_control=start,commit_local=local,plan_accepted_update=accepted,
                target_reconstruction_bitexact=True,gain_is_zero=bool(not np.any(gain)),
                tangent_max_abs=float(np.abs(tangent).max()),raw_feedback_max_abs_rad=float(np.abs(raw).max()),
                feedback_clipped_components=int(np.sum(raw!=correction)),target_clipped_components=int(np.sum(preclip!=target)),
                feedback_boundary_components=int(feedback_boundary.sum()),target_boundary_components=int(target_boundary.sum()),
                velocity_derivative_undefined_output_rows=[names[i] for i in np.flatnonzero(nondiff)],
                velocity_derivative_smooth_active_output_rows=int(smooth_active.sum()),velocity_derivative_smooth_zero_output_rows=int(valid_zero.sum()),
                joint_velocity_K_max_abs=float(np.abs(v).max()),joint_velocity_K_frobenius=float(np.linalg.norm(v)),
                joint_velocity_K_spectral_norm=float(np.linalg.norm(v,2)),
                known_smooth_velocity_J_spectral_norm=float(np.linalg.norm(smooth_v,2)),
                joint_position_K_max_abs=float(np.abs(q).max()),joint_position_K_spectral_norm=float(np.linalg.norm(q,2)),
                velocity_K_diagonal=diag.tolist(),velocity_K_diag_positive=int(np.sum(diag>1e-12)),
                velocity_K_diag_negative=int(np.sum(diag< -1e-12)),velocity_K_diag_zero=int(np.sum(np.abs(diag)<=1e-12)),
                smallest_interior_single_joint_velocity_axis_radius_radps=axis_radius,
                some_001radps_velocity_axis_feedback_change_above_01rad=bool(np.abs(v).max()*.01>.1))
            rows.append(row)
        allrows+=rows
        reports[name]=dict(rows=len(rows),controls=[250,1268],all_actual_targets_reconstructed_bitexact=True,
            feedback_clipped_rows=sum(r['feedback_clipped_components']>0 for r in rows),
            feedback_clipped_components=sum(r['feedback_clipped_components'] for r in rows),
            native_target_clipped_rows=sum(r['target_clipped_components']>0 for r in rows),
            native_target_clipped_components=sum(r['target_clipped_components'] for r in rows),
            velocity_derivative_nondifferentiable_controls=sum(bool(r['velocity_derivative_undefined_output_rows']) for r in rows),
            velocity_derivative_nondifferentiable_components=sum(len(r['velocity_derivative_undefined_output_rows']) for r in rows),
            zero_gain_controls=[r['control'] for r in rows if r['gain_is_zero']],
            no_accepted_update_controls=[r['control'] for r in rows if r['plan_accepted_update'] is False],
            joint_velocity_max_abs_K=summary([r['joint_velocity_K_max_abs'] for r in rows]),
            joint_velocity_spectral_K=summary([r['joint_velocity_K_spectral_norm'] for r in rows]),
            known_smooth_velocity_spectral_J=summary([r['known_smooth_velocity_J_spectral_norm'] for r in rows]),
            joint_position_spectral_K=summary([r['joint_position_K_spectral_norm'] for r in rows]),
            raw_actual_feedback_max_rad=max(r['raw_feedback_max_abs_rad'] for r in rows),
            velocity_diagonal_sign_counts={sign:sum(r['velocity_K_diag_'+sign] for r in rows) for sign in ('positive','negative','zero')},
            controls_with_001radps_axis_linear_feedback_change_above_01=sum(r['some_001radps_velocity_axis_feedback_change_above_01rad'] for r in rows),
            special_first_five=[r for r in rows if r['control']<255])
    # Exact direct feature insertion for joint q/dq at fixed BFM base and prior.
    result=dict(kind='three_expert_frozen_feedback_arithmetic_audit',total_moving_rows=len(allrows),datasets=reports,
        native_tangent_order='roottranslation0:3; rightmultiplicative bodyrotation3:6; jointposition6:29; rawvelocity29:58 with jointvelocity35:58',
        feedback_sign='K already contains negative backward-solve sign; difference(actual,nominal) is implemented y_actual minus x_nominal, with conjugate(q_nominal)*q_actual for orientation.',
        command='native_clip(planned_target + clip(K*difference(planned_state,actual_state),-.1,.1))',
        joint_velocity_feature_mapping=dict(physical_velocity_columns=[35,58],direct_feature_columns=[23,46],
            BFM_state_columns=[23,46],BFM_base_feature_columns=[1023,1046],history_constant=True,previous_action_constant=True,reference_goal_constant=True),
        joint_position_feature_mapping=dict(physical_tangent_columns=[6,29],direct_feature_columns=[0,23],BFM_state_columns=[0,23],BFM_base_feature_columns=[1023,1046]),
        exact_derivative_scope='Frozen already-committed local feedback at recorded expert states, excluding clipping kink directions; not a derivative of fresh MPC replanning.',
        BFM_base_derivative_measured=False,training_derivative_labels_generated=False,augmented_target_labels_generated=False,
        physical_perturbations=0,inference_calls=0,optimizer_calls=0,physics_steps=0,hardware_authorized=False)
    (BASE/'per_control.json').write_text(json.dumps(allrows,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    result['per_control_sha256']=sha(BASE/'per_control.json')
    (BASE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (BASE/'input_hashes.json').write_text(json.dumps(paths,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in data.items() if k!='special_first_five'} for name,data in reports.items()},indent=2))

if __name__=='__main__':main()
