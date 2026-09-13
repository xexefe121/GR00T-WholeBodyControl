"""Analytical frozen-head sensitivity on saved inputs only. No BFM inference/dynamics/optimizer."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,hashlib
import numpy as np
import torch
BASE=Path(r'E:/codex-artifacts/sonic23_teleop_resume_20260911')
RUN=BASE/'fast_controller_phase_fit_v1'
OUT=Path(__file__).resolve().parent
CONTROLS=[250,251,252,255,260,270,278]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def arc(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def norm2(x):return float(np.linalg.svd(x,compute_uv=False)[0])
def summary(x):return dict(rms=rms(x),maximum_abs=float(np.max(np.abs(x))))
trace=arc(RUN/'nominal/trace.npz');labels=arc(BASE/'bfm_entry250_labels_v1/labels/labels.npz');fit=arc(RUN/'fit/teacher_fit.npz')
assert sha(RUN/'nominal/trace.npz')=='bec4809cdb19c9b153d726485b7a77d2640c7bd399362681801bbd6abad086b9'
saved=torch.load(RUN/'fit/student_head.pt',map_location='cpu',weights_only=True)
assert saved['completed_steps']==65000 and sha(RUN/'fit/student_head.onnx')=='861b4c39349276851e23edab43978a74bab4cc613915da87805471c6164bc365'
mean,std,span=[saved[k].numpy().astype(np.float64) for k in ('feature_mean','feature_std','joint_span')]
weights=[saved['actor_state']['%d.weight'%k].numpy().astype(np.float64) for k in (0,2,4)]
biases=[saved['actor_state']['%d.bias'%k].numpy().astype(np.float64) for k in (0,2,4)]
def elu(x):return np.where(x>0,x,np.expm1(np.minimum(x,0)))
def forward(x):
 a=(x-mean)/std
 for w,b in zip(weights[:2],biases[:2]):a=elu(a@w.T+b)
 return (a@weights[2].T+biases[2])*span
def jacobian(x):
 a0=weights[0]@((x-mean)/std)+biases[0]
 a1=weights[1]@elu(a0)+biases[1]
 d0=np.where(a0>0,1.,np.exp(np.minimum(a0,0)))
 d1=np.where(a1>0,1.,np.exp(np.minimum(a1,0)))
 return ((((span[:,None]*weights[2])*d1[None,:])@weights[1])*d0[None,:])@weights[0]/std[None,:]
contract_path=Path(r'Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
c=json.loads(contract_path.read_text());limits=np.asarray(c['joint_limits']);C=.25*np.asarray(c['training_effort'])/np.asarray(c['kp']);S=1/C;default=np.asarray(c['default_q'])
blocks={'joint_position':(0,23),'joint_velocity':(23,46),'root_angular_velocity':(46,49),'gravity':(49,52),'previous_target':(52,75),'root_linear_velocity':(75,78),'root_height':(78,79),'received_goals_in_actual_heading':(79,1023),'BFM_base':(1023,1046),'previous_raw_action':(1046,1069)}
assert sum(stop-start for start,stop in blocks.values())==1069
rows=[];arrays={};points,qweights=np.polynomial.legendre.leggauss(64);points=(points+1)/2;qweights=qweights/2
for control in CONTROLS:
 row=control-250;x=trace['features'][control].astype(np.float64);reference=labels['features'][row].astype(np.float64)
 J=jacobian(x);teacher_J=jacobian(reference);prediction=forward(x);teacher_prediction=forward(reference)
 actual_delta=trace['delta'][control];stored_teacher_delta=fit['predicted_delta'][2038+row]
 actual_max=float(np.max(np.abs(prediction-actual_delta)));teacher_max=float(np.max(np.abs(teacher_prediction-stored_teacher_delta)))
 assert actual_max<1e-5 and teacher_max<1e-5,(control,actual_max,teacher_max)
 # Central directional derivative checks use mathematical float64 head, no model runtime.
 directions=np.eye(1069)[[0,23,52,74,79,1023,1046,1068]]
 eps=1e-6
 numeric=np.stack([(forward(x+eps*d)-forward(x-eps*d))/(2*eps) for d in directions],axis=1)
 analytical=J@directions.T;derivative_max=float(np.max(np.abs(numeric-analytical)));assert derivative_max<1e-7
 previous=trace['previous_action'][control].astype(np.float64);previous_unclipped=default+previous*C
 prev_mask=((previous_unclipped>limits[:,0])&(previous_unclipped<limits[:,1])).astype(float)
 direct_previous_target=J[:,52:75]*prev_mask[None,:]
 direct_previous_raw=J[:,1046:1069]*S[None,:]
 Rtarget=direct_previous_target+direct_previous_raw
 Rraw=S[:,None]*Rtarget*C[None,:]
 B=np.eye(23)+J[:,1023:1046]
 unclipped=trace['base_target'][control]+prediction;active=((unclipped>limits[:,0])&(unclipped<limits[:,1])).astype(float)
 d=x-reference;delta_difference=prediction-teacher_prediction
 for quadrature_points in (64,128,256,512,1024,2048):
  points,qweights=np.polynomial.legendre.leggauss(quadrature_points);points=(points+1)/2;qweights=qweights/2
  average_J=np.zeros_like(J)
  for q,w in zip(points,qweights):average_J+=w*jacobian(reference+q*d)
  block_contributions={name:average_J[:,a:b]@d[a:b] for name,(a,b) in blocks.items()}
  integrated_sum=sum(block_contributions.values());closure=float(np.max(np.abs(integrated_sum-delta_difference)))
  if closure<1e-5:break
 assert closure<1e-5,(control,closure)
 base_change=trace['base_target'][control]-labels['base_target'][row]
 actual_target=np.clip(unclipped,limits[:,0],limits[:,1]);teacher_target=np.clip(labels['base_target'][row]+teacher_prediction,limits[:,0],limits[:,1])
 applied_history=(trace['target'][control]-default)*S
 result=dict(control=control,teacher_row=row,mathematical_float64_vs_recorded_ONNX_max_rad=actual_max,
  mathematical_float64_vs_stored_teacherfit_max_rad=teacher_max,analytical_derivative_central_difference_max=derivative_max,
  normalized_feature_departure_rms=rms(d/std),head_delta_departure_rad=summary(delta_difference),BFM_base_departure_rad=summary(base_change),
  target_change_from_teacherstate_prediction_rad=summary(actual_target-teacher_target),
  teacherstate_fit_error_rad=summary(teacher_target-labels['expert_target'][row]),
  actual_target_minus_sameclock_teacher_target_rad=summary(trace['target'][control]-labels['expert_target'][row]),
  previous_target_clipped_components=int(np.sum(prev_mask==0)),current_target_clipped_components=int(np.sum(active==0)),
  preclip_history_minus_applied_target_history=summary(trace['action'][control]-applied_history),
  direct_previous_action_feedback_conditional=dict(target_equivalent_spectral_norm=norm2(Rtarget),raw_action_spectral_norm=norm2(Rraw),
   spectral_radius=float(np.max(np.abs(np.linalg.eigvals(Rtarget)))),previous_target_path_norm=norm2(direct_previous_target),raw_action_path_norm=norm2(direct_previous_raw),
   clipped_physical_target_response_norm=norm2(active[:,None]*Rtarget),BFM_base_fixed=True,plant_feedback_omitted=True,not_closed_loop_stability_eigenvalues=True),
  conditional_BFM_base_to_unclipped_target_norm=norm2(B),
  feature_block_derivative_norm={name:norm2(J[:,a:b]) for name,(a,b) in blocks.items()},
  integrated_sameclock_teacher_to_actual=dict(quadrature_points=quadrature_points,closure_max_rad=closure,
   block_contribution_rad={name:value.tolist() for name,value in block_contributions.items()},
   block_rms_rad={name:rms(value) for name,value in block_contributions.items()},BFM_direct_identity_contribution_rad=base_change.tolist(),
   explanation='Head-output difference along fixed straight line in feature space; correlated blocks may cancel. Savedbase direct identity adds separately. Off-manifold path, not causal dynamics proof.'))
 rows.append(result)
 arrays.update({f'control{control}_J_raw_features':J,f'control{control}_teacher_J_raw_features':teacher_J,
  f'control{control}_conditional_prior_target_jacobian':Rtarget,f'control{control}_conditional_prior_raw_jacobian':Rraw,
  f'control{control}_conditional_base_jacobian':B,f'control{control}_integrated_J':average_J,
  f'control{control}_actual_features':x,f'control{control}_teacher_features':reference,
  f'control{control}_mathematical_head_output':prediction})
np.savez_compressed(OUT/'jacobians.npz',**arrays)
paths=[RUN/'nominal/trace.npz',RUN/'nominal/report.json',RUN/'actual_query250_input_parity.json',RUN/'fit/student_head.pt',RUN/'fit/student_head.onnx',RUN/'fit/teacher_fit.npz',BASE/'bfm_entry250_labels_v1/labels/labels.npz',RUN/'source_snapshot_v1/student_linear_runtime.py',RUN/'source_snapshot_v1/gear_sonic/utils/g1_true23_mpc_student.py',contract_path,Path(__file__)]
report=dict(kind='analytical_frozen_head_sensitivity_saved_inputs',controls=CONTROLS,feature_blocks=blocks,rows=rows,
 analytical_head_evaluations_only=True,BFM_inference_calls=0,optimizer_calls=0,physics_steps=0,new_candidates=0,
 derivative_convention='Smooth float64 evaluation of fixed stored float32 weights; output radians/input raw feature units. Validated against savedONNXoutputs and central differences.',
 limits=['Full300history is not direct head input; history and proprio effects on BFM cannot be separated using savedbase alone.',
 'Conditional prior recurrence freezes BFM, goals and physical state; it does not measure complete closed-loop stability.',
 'Actual versus sameclock expertstate command differences are not actualstate expert errors; new counterfactual expert was not queried.',
 'Input-block straightline decomposition is descriptive; independently changing correlated feature blocks can leave realizable state manifold.'],
 input_sha256={str(p):sha(p) for p in paths},jacobians_sha256=sha(OUT/'jacobians.npz'))
write('report.json',report)
print(json.dumps([dict(control=r['control'],prior_norm=r['direct_previous_action_feedback_conditional']['target_equivalent_spectral_norm'],prior_radius=r['direct_previous_action_feedback_conditional']['spectral_radius'],base_norm=r['conditional_BFM_base_to_unclipped_target_norm'],feature_departure=r['normalized_feature_departure_rms'],head_departure=r['head_delta_departure_rad']['rms'],base_departure=r['BFM_base_departure_rad']['rms'],blocks=r['integrated_sameclock_teacher_to_actual']['block_rms_rad']) for r in rows]))
