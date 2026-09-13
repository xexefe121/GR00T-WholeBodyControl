"""One frozen232-case parity/cost benchmark; no connected policy or fallback."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time,importlib.util
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
sys.path.insert(0,str(BASE/'preserved_walk_demo_v1/repo'))
import numpy as np
import mujoco
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle
from native_forecast import NativeForecast
spec=importlib.util.spec_from_file_location('frozen_independent_oracle',HERE/'oracle_snapshot.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def path(p):
 s=str(p).replace('\\','/')
 return Path('/mnt/'+s[0].lower()+s[2:]) if len(s)>2 and s[1]==':' else Path(s)
def exact(a,b,name):
 a,b=np.asarray(a),np.asarray(b)
 assert a.shape==b.shape and a.dtype==b.dtype,(name,'shape/dtype',a.shape,b.shape,a.dtype,b.dtype)
 assert a.tobytes()==b.tobytes(),name

def main():
 request=json.loads((HERE/'request.json').read_text())
 for p,digest in request['input_sha256'].items():assert sha(path(p))==digest,p
 assert request['case_count']==232 and request['connected_controller'] is False
 with np.load(HERE/'cases.npz',allow_pickle=False) as a:cases={k:a[k].copy() for k in a.files}
 for a in cases.values():a.flags.writeable=False
 native,contract,*_=load_native_bundle(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1','walk003')
 spec=mujoco.mjtState.mjSTATE_INTEGRATION;source=mujoco.MjData(native);engine=NativeForecast(native,contract)
 assert mujoco.mj_stateSize(native,spec)==291
 frozen_rows=json.loads((BASE/'phase_student_private_forecasts_v1/report.json').read_text())['all_forecasts']
 output=HERE/'results';output.mkdir(exist_ok=False)
 shape={'physics_qpos':(51,30),'physics_qvel':(51,29),'physics_time':(51,),
  'physics_torque':(50,23),'physics_actuator_force':(50,23),'warning_counts':(51,8),'warning_lastinfo':(51,8)}
 storage={}
 for name in ('preallocated','oracle'):
  storage[name]={k:np.zeros((232,*v),dtype=np.int32 if k.startswith('warning') else np.float64) for k,v in shape.items()}
 rows=[];completed=0;roundtrip=np.empty(291);source_before=np.empty(291);source_after=np.empty(291);state=np.empty(291)
 try:
  for i in range(232):
   np.copyto(state,cases['integration'][i]);target=cases['target'][i];controls=int(cases['horizon_controls'][i])
   mujoco.mj_setState(native,source,state,spec);source.warning.number[:]=cases['warning_counts'][i];source.warning.lastinfo[:]=cases['warning_lastinfo'][i]
   mujoco.mj_forward(native,source);mujoco.mj_setState(native,source,state,spec)
   source.warning.number[:]=cases['warning_counts'][i];source.warning.lastinfo[:]=cases['warning_lastinfo'][i]
   mujoco.mj_getState(native,source,roundtrip,spec);exact(roundtrip,state,'source initial291')
   exact(source.qpos,cases['qpos'][i],'source qpos');exact(source.qvel,cases['qvel'][i],'source qvel')
   mujoco.mj_getState(native,source,source_before,spec)
   results={};timings={};order=('preallocated','oracle') if i%2==0 else ('oracle','preallocated')
   constant_targets=np.tile(target,(controls,1))
   for name in order:
    tick=time.perf_counter_ns()
    report,trace=(engine.predict(source,target,controls) if name=='preallocated' else
     oracle.inspect_native_segment(native,source,constant_targets,contract,stop_on_failure=True,retain_trace=True))
    for k,a in trace.items():storage[name][k][i,:len(a)]=a
    timings[name]=(time.perf_counter_ns()-tick)/1e6
    results[name]=(report,trace)
    mujoco.mj_getState(native,source,source_after,spec);exact(source_after,source_before,'caller291 unchanged '+name)
    exact(source.warning.number,cases['warning_counts'][i],'caller warnings unchanged '+name)
    exact(source.warning.lastinfo,cases['warning_lastinfo'][i],'caller warning info unchanged '+name)
   a,b=results['preallocated'],results['oracle']
   for key in shape:exact(a[1][key],b[1][key],f'case{i} every sample {key}')
   for key in ('feasible','first_failure','physics_steps','maximum','minimum_root_height_m','initial_time','final_time','final_warning_counts','original_data_unchanged'):
    assert a[0][key]==b[0][key],(i,key,a[0][key],b[0][key])
   original=frozen_rows[i]
   assert original['control']==int(cases['control'][i]) and original['candidate']==str(cases['candidate'][i]) and original['constant_target_controls']==controls
   for key,newkey in [('feasible','feasible'),('first_failure','first_failure'),('maximum','maximum'),('checked_steps','physics_steps')]:assert original[key]==b[0][newkey],(i,key)
   rows.append(dict(case=i,control=int(cases['control'][i]),candidate=str(cases['candidate'][i]),horizon_controls=controls,
    first_failure=a[0]['first_failure'],feasible=a[0]['feasible'],physics_steps=a[0]['physics_steps'],all_seven_trace_fields_bitexact=True,
    report_fields_exact=True,root_saved_forecast_report_exact=True,caller291_warnings_unchanged=True,timing_ms=timings,timing_order=list(order)))
   completed=i+1
   if completed%16==0:print(json.dumps(dict(completed_cases=completed,latest_fast_ms=timings['preallocated'],latest_oracle_ms=timings['oracle'])),flush=True)
 except Exception as error:
  np.savez_compressed(output/'partial_traces.npz',**{name+'_'+k:a[:completed+1] for name,fields in storage.items() for k,a in fields.items()})
  (output/'failure.json').write_text(json.dumps(dict(completed_cases=completed,exception_type=type(error).__name__,message=str(error),rows=rows),indent=2)+'\n')
  raise
 assert completed==engine.forecasts==232
 for name,fields in storage.items():np.savez_compressed(output/(name+'_traces.npz'),**fields,checked_steps=np.asarray([r['physics_steps'] for r in rows]))
 summary={}
 for controls in (1,5):
  selected=[r for r in rows if r['horizon_controls']==controls]
  summary[str(controls)]={name:np.percentile([r['timing_ms'][name] for r in selected],[50,95,100]).tolist() for name in ('preallocated','oracle')}
 for p,digest in request['input_sha256'].items():assert sha(path(p))==digest,p
 result=dict(kind='fixed232_private_preallocated_forecast_parity_and_cost',pass_=True,cases=232,
  requested_horizons_controls=[1,5],same_fixed_root_cases=True,all_state_torque_force_time_warning_samples_bitexact=True,
  all_first_failures_and_metrics_exact=True,all_root_saved_forecast_rows_exact=True,all_caller_integration_and_warnings_unchanged=True,
  no_per_call_MjData_or_trace_workspace_allocation=True,per_call_python_metadata_and_views_allocated=True,
  no_deepcopy_in_component=True,private_forward_then_full_state_restoration=True,
  timing_ms_p50_p95_max=summary,timing_includes='fullstate/warning copy,privateforward+restore,allstrictchecks,fulltrace workspace writes,result metadata,owned output storage copy,component caller-immutability checks',
  benchmark_order='fixed alternating component/oracle order, one pass percase, no bestof selection',
  contention_uncontrolled=True,real_time_qualified=False,no_connected_controller=True,no_fallback_selection=True,
  policy_inference_calls=0,optimizer_calls=0,actual_plant_steps=0,
  private_component_steps=sum(r['physics_steps'] for r in rows),private_oracle_steps=sum(r['physics_steps'] for r in rows),rows=rows,
  request_sha256=sha(HERE/'request.json'),output_sha256={name:sha(output/name) for name in ('preallocated_traces.npz','oracle_traces.npz')})
 (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ('pass_','cases','private_component_steps','private_oracle_steps','timing_ms_p50_p95_max')}),flush=True)
if __name__=='__main__':main()
