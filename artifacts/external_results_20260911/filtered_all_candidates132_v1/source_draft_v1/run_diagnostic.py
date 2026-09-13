"""Evaluate exactly132 fixed private forecasts and original cost components."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import mujoco
from native_forecast import NativeForecast
import gear_sonic.utils.g1_true23_mjbatch_mpc as cost_module
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker,load_native_bundle,load_motion_override,TRACKED
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy

BASE=Path(__file__).resolve().parent.parent;NEW=BASE.parent
ROOT=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
REFERENCE=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz')
FILTER=NEW/'fast_controller_filtered_v1'
NAMES=('primary','original_BFM','previous_applied','current_q')
COMPONENTS=('world_body_position','body_rotation_log','joint_position','generalized_velocity','lower_joint_margin','upper_joint_margin','native_speed_soft_margin')
SLICES=(slice(0,18),slice(18,36),slice(36,59),slice(59,88),slice(88,111),slice(111,134),slice(134,157))

def path(value):
    value=str(value).replace('\\','/')
    return Path('/mnt/'+value[0].lower()+value[2:]) if len(value)>2 and value[1]==':' else Path(value)
def sha(value):return hashlib.sha256(Path(value).read_bytes()).hexdigest()
def read(value):return json.loads(Path(value).read_text())
def load(value):
    with np.load(value,allow_pickle=False) as data:return {key:data[key].copy() for key in data.files}
def plain(value):
    if isinstance(value,dict):return {str(key):plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [plain(item) for item in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
def write(value,data):Path(value).write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n',encoding='utf-8')
def exact(actual,wanted,name):
    actual,wanted=np.asarray(actual),np.asarray(wanted)
    assert actual.dtype==wanted.dtype and actual.shape==wanted.shape,(name,actual.dtype,wanted.dtype,actual.shape,wanted.shape)
    assert actual.tobytes()==wanted.tobytes(),name
def yaw(q):
    return np.arctan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2]**2+q[...,3]**2))

def main():
    request=read(BASE/'request.json')
    assert request['case_count']==request['private_forecast_calls']==132 and not request['connected_controller']
    for name,digest in request['source_sha256'].items():assert sha(BASE/'source_snapshot_v1'/name)==digest,name
    for value,digest in request['input_sha256'].items():assert sha(path(value))==digest,value
    assert Path(cost_module.__file__).resolve()==BASE/'source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_mpc.py'
    assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    cases=load(BASE/'cases.npz');old=load(FILTER/'nominal/private_forecasts.npz')
    old_lookup=dict(request['existing_saved_forecast_comparisons'])
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,manifest)
    source=mujoco.MjData(native);fk=mujoco.MjData(native);engine=NativeForecast(native,c,maximum_controls=5)
    # Constructor/forward features only: preserve the qualified H30 tracker and all original weights.
    tracker=Native23Tracker(position_servo_copy(native,np.asarray(c['kp']),np.asarray(c['kd']),np.asarray(c['native_effort'])),
        c,motion,horizon=30,threads=2,all_joint_limit_margin=.05,all_joint_limit_weight=2000,
        relative_foot_weight=0,fd_epsilon=1e-6,hard_feasibility=True)
    assert tracker.T==30 and tracker.relative_foot_weight==0 and tracker.limit_weight==2000 and tracker.limit_margin==.05
    def forbidden(*args,**kwargs):raise RuntimeError('No planner simulation, optimization, or derivative calculation is authorized.')
    for name in ('rollout','advance','step','linearize','expand'):setattr(tracker,name,forbidden)
    native_ids=[native.body(name).id for name in TRACKED]
    np.testing.assert_array_equal(native_ids,tracker.ids)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(291);roundtrip=np.empty(291)
    shape=dict(physics_qpos=(51,30),physics_qvel=(51,29),physics_time=(51,),physics_torque=(50,23),
        physics_actuator_force=(50,23),warning_counts=(51,8),warning_lastinfo=(51,8))
    traces={key:np.zeros((132,*dims),np.int32 if key.startswith('warning_') else np.float64) for key,dims in shape.items()}
    knots=dict(knot_available=np.zeros((132,6),bool),state=np.full((132,6,59),np.nan),features=np.full((132,6,101),np.nan),
        residual=np.full((132,6,157),np.nan),component_cost=np.full((132,6,7),np.nan),state_cost=np.full((132,6),np.nan),
        input_cost=np.full((132,5),np.nan),target_reference=np.full((132,5,23),np.nan),
        root_position_error=np.full((132,6,3),np.nan),root_height_error=np.full((132,6),np.nan),
        root_vertical_velocity=np.full((132,6),np.nan),root_vertical_velocity_error=np.full((132,6),np.nan),
        root_yaw_abs_deg=np.full((132,6),np.nan),joint_error=np.full((132,6,23),np.nan),joint_rmse=np.full((132,6),np.nan),
        feet_world_position_error=np.full((132,6,2),np.nan),feet_root_relative_position_error=np.full((132,6,2),np.nan),
        world_position_component_by_body_axis=np.full((132,6,6,3),np.nan),body_rotation_component=np.full((132,6,6),np.nan),
        generalized_velocity_component=np.full((132,6,29),np.nan))
    output=BASE/'results';output.mkdir(exist_ok=False)
    completed=0;rows=[];old_comparisons=0;direct_cost_checks=0;native_fk_checks=0;component_sum_max_error=0.;started=time.perf_counter()
    try:
        for case in range(132):
            control=int(cases['control'][case]);candidate=int(cases['candidate'][case]);target=cases['target'][case]
            np.copyto(state,cases['integration'][case]);mujoco.mj_setState(native,source,state,spec)
            source.warning.number[:]=cases['warning_counts'][case];source.warning.lastinfo[:]=cases['warning_lastinfo'][case]
            mujoco.mj_forward(native,source);mujoco.mj_setState(native,source,state,spec)
            source.warning.number[:]=cases['warning_counts'][case];source.warning.lastinfo[:]=cases['warning_lastinfo'][case]
            mujoco.mj_getState(native,source,roundtrip,spec);exact(roundtrip,state,'initial291')
            exact(source.qpos,cases['qpos'][case],'initial qpos');exact(source.qvel,cases['qvel'][case],'initial qvel')
            tick=time.perf_counter();report,views=engine.predict(source,target,horizon_controls=5)
            # Save expiring buffers before any later prediction. Only one predict call per fixed case.
            for key,value in views.items():traces[key][case,:len(value)]=value
            forecast_ms=(time.perf_counter()-tick)*1000
            mujoco.mj_getState(native,source,roundtrip,spec);exact(roundtrip,state,'caller291 unchanged')
            exact(source.warning.number,cases['warning_counts'][case],'caller warnings unchanged')
            exact(source.warning.lastinfo,cases['warning_lastinfo'][case],'caller warninginfo unchanged')
            steps=report['physics_steps'];count=steps//10+1
            if case in old_lookup:
                index=old_lookup[case];ss,se=old['state_offsets'][index:index+2];ts,te=old['step_offsets'][index:index+2]
                assert se-ss==steps+1 and te-ts==steps
                for key in shape:
                    prior=old[key][ts:te] if key in ('physics_torque','physics_actuator_force') else old[key][ss:se]
                    exact(traces[key][case,:len(prior)],prior,'previous saved candidate '+key)
                old_comparisons+=1
            predicted=np.concatenate((traces['physics_qpos'][case,np.arange(count)*10],traces['physics_qvel'][case,np.arange(count)*10]),axis=1)
            tracker.window(control+10);features=tracker.features(predicted)
            assert features.shape==(count,101)
            # Geometry evaluated with the unchanged native model must match the servo copy's cost features.
            for knot,predicted_state in enumerate(predicted):
                fk.qpos[:]=predicted_state[:30];fk.qvel[:]=predicted_state[30:];mujoco.mj_kinematics(native,fk)
                exact(features[knot,:18],fk.xpos[native_ids].reshape(-1),'native world body positions')
                exact(features[knot,18:42],fk.xquat[native_ids].reshape(-1),'native body quaternions');native_fk_checks+=1
            residual=tracker.residual(np.arange(count),features)
            assert residual.shape==(count,157)
            squared=residual**2;components=np.stack([np.sum(squared[:,sl],axis=-1) for sl in SLICES],axis=1)
            state_cost=np.sum(squared,axis=1);target_ref=tracker.target_reference(np.arange(5))
            input_cost=tracker.control_weight*np.sum((target[None]-target_ref)**2,axis=1)
            component_sum_max_error=max(component_sum_max_error,float(np.max(np.abs(np.sum(components,axis=1)-state_cost))))
            # Direct calls to the original cost method verify each available original running term.
            for knot in range(min(count,5)):
                tracker.window(control+10+knot)
                direct=tracker.cost(0,predicted[knot:knot+1],target[None])[0]
                exact(np.asarray(direct),np.asarray(state_cost[knot]+input_cost[knot]),'original cost direct evaluation');direct_cost_checks+=1
            tracker.window(control+10)
            ref=tracker.reference[control+10+np.arange(count)];positions=features[:,:18].reshape(count,6,3);refpos=ref[:,:18].reshape(count,6,3)
            refstate=ref[:,42:];joint_error=predicted[:,7:30]-refstate[:,7:30]
            yaw_error=np.angle(np.exp(1j*(yaw(predicted[:,3:7])-yaw(refstate[:,3:7]))))
            values=dict(knot_available=np.ones(count,bool),state=predicted,features=features,residual=residual,component_cost=components,
                state_cost=state_cost,root_position_error=positions[:,0]-refpos[:,0],root_height_error=positions[:,0,2]-refpos[:,0,2],
                root_vertical_velocity=predicted[:,32],root_vertical_velocity_error=predicted[:,32]-refstate[:,32],
                root_yaw_abs_deg=np.abs(yaw_error)*180/np.pi,joint_error=joint_error,joint_rmse=np.sqrt(np.mean(joint_error**2,axis=1)),
                feet_world_position_error=np.linalg.norm(positions[:,2:4]-refpos[:,2:4],axis=-1),
                feet_root_relative_position_error=np.linalg.norm((positions[:,2:4]-positions[:,0:1])-(refpos[:,2:4]-refpos[:,0:1]),axis=-1),
                world_position_component_by_body_axis=squared[:,:18].reshape(count,6,3),body_rotation_component=squared[:,18:36].reshape(count,6,3).sum(axis=-1),
                generalized_velocity_component=squared[:,59:88])
            for key,value in values.items():knots[key][case,:count]=value
            knots['input_cost'][case]=input_cost;knots['target_reference'][case]=target_ref
            available_complete=count==6;prefix_sum=float(np.sum(state_cost)+np.sum(input_cost)) if available_complete else None
            eligible=bool(report['feasible'] and steps==50)
            final=traces['physics_qpos'][case,steps];finalv=traces['physics_qvel'][case,steps]
            rows.append(dict(case=case,control=control,candidate=candidate,candidate_name=NAMES[candidate],**report,
                forecast_ms=forecast_ms,available_state_knots=count,postcontrol_goal_frames=(control+10+np.arange(1,count)).tolist(),
                full100ms_cost_available=available_complete,eligible_for_fixed_horizon_comparison=eligible,
                initial_state_cost=float(state_cost[0]),five_control_prefix_state_and_input_sum=prefix_sum,
                postcontrol_state_cost_sum=float(np.sum(state_cost[1:])) if available_complete else None,
                five_input_cost_sum=float(np.sum(input_cost)),
                full_prefix_component_sums={name:float(np.sum(components[:,i])) for i,name in enumerate(COMPONENTS)} if available_complete else None,
                end_of_checked_forecast_root_height_m=float(final[2]),end_of_checked_forecast_vertical_velocity_mps=float(finalv[2]),
                most_negative_checked_vertical_velocity_mps=float(np.min(traces['physics_qvel'][case,:steps+1,2])),
                endpoint100ms=dict(root_height_error_m=float(values['root_height_error'][-1]),
                    vertical_velocity_mps=float(values['root_vertical_velocity'][-1]),
                    vertical_velocity_error_mps=float(values['root_vertical_velocity_error'][-1]),
                    root_position_error_norm_m=float(np.linalg.norm(values['root_position_error'][-1])),
                    root_yaw_abs_deg=float(values['root_yaw_abs_deg'][-1]),joint_rmse_rad=float(values['joint_rmse'][-1]),
                    feet_world_error_m=values['feet_world_position_error'][-1],feet_root_relative_error_m=values['feet_root_relative_position_error'][-1]) if available_complete else None,
                existing_saved_forecast_bitexact=case in old_lookup))
            completed+=1
            if completed%16==0:print(json.dumps(dict(completed_cases=completed,private_steps=sum(row['physics_steps'] for row in rows))),flush=True)
    except Exception as error:
        np.savez_compressed(output/'partial_forecasts.npz',**{key:value[:completed+1] for key,value in traces.items()})
        np.savez_compressed(output/'partial_cost_knots.npz',**{key:value[:completed+1] for key,value in knots.items()})
        write(output/'failure.json',dict(completed_cases=completed,exception_type=type(error).__name__,message=str(error),rows=rows));raise
    assert completed==engine.forecasts==132 and old_comparisons==50
    np.savez_compressed(output/'forecasts.npz',**traces,checked_steps=np.asarray([row['physics_steps'] for row in rows]),control=cases['control'],candidate=cases['candidate'],target=cases['target'])
    np.savez_compressed(output/'cost_knots.npz',**knots,control=cases['control'],candidate=cases['candidate'],
        state_goal_frames=cases['control'][:,None]+10+np.arange(6),input_goal_frames=cases['control'][:,None]+11+np.arange(5))
    comparisons=[]
    for control in range(250,283):
        group=rows[(control-250)*4:(control-249)*4];eligible=[row for row in group if row['eligible_for_fixed_horizon_comparison']]
        ordered=sorted(eligible,key=lambda row:(row['five_control_prefix_state_and_input_sum'],row['candidate']))
        primary=group[0];best=ordered[0] if ordered else None
        comparisons.append(dict(control=control,eligible_candidates=[row['candidate_name'] for row in eligible],
            cost_order=[row['candidate_name'] for row in ordered],lowest_cost_candidate=None if best is None else best['candidate_name'],
            primary_feasible=primary['feasible'],primary_prefix_cost=primary['five_control_prefix_state_and_input_sum'],
            best_prefix_cost=None if best is None else best['five_control_prefix_state_and_input_sum'],
            primary_excess_over_best=None if best is None or not primary['eligible_for_fixed_horizon_comparison'] else primary['five_control_prefix_state_and_input_sum']-best['five_control_prefix_state_and_input_sum']))
    for name,digest in request['source_sha256'].items():assert sha(BASE/'source_snapshot_v1'/name)==digest
    for value,digest in request['input_sha256'].items():assert sha(path(value))==digest
    result=dict(kind=request['kind'],completed_cases=132,all132_outcomes_retained=True,private_forecast_calls=engine.forecasts,
        private_physics_steps=sum(row['physics_steps'] for row in rows),actual_plant_steps=0,actor_calls=0,optimizer_calls=0,
        original_H30_tracker_features_only=True,planner_rollout_or_step_calls=0,existing50_forecasts_all_seven_fields_bitexact=True,
        all132_source_full291_and_warning_states_unchanged=True,original_cost_direct_running_term_checks=direct_cost_checks,
        native_model_and_servo_cost_geometry_bitexact_checks=native_fk_checks,component_sum_roundoff_max=component_sum_max_error,
        component_order=list(COMPONENTS),tracked_bodies=list(TRACKED),cost_conventions=request['cost_scope'],
        state_frame_convention=request['state_goal_frames'],input_frame_convention=request['target_goal_frames'],
        no_terminal_multiplier=True,no_horizon_rescaling=True,full_H30_objective_not_evaluated=True,
        interrupted_forecasts_are_censored=True,diagnostic_cost_ranking_never_applied_to_actual_plant=True,
        counterscenarios_are_separate_saved_states_not_a_recovered_trajectory=True,
        elapsed_seconds=time.perf_counter()-started,rows=rows,control_comparisons=comparisons,
        request_sha256=sha(BASE/'request.json'),output_sha256={name:sha(output/name) for name in ('forecasts.npz','cost_knots.npz')},
        motion_override=override,hardware_authorized=False)
    write(output/'report.json',result)
    print(json.dumps({key:result[key] for key in ('completed_cases','private_physics_steps','original_cost_direct_running_term_checks','existing50_forecasts_all_seven_fields_bitexact','elapsed_seconds')}),flush=True)

if __name__=='__main__':main()
