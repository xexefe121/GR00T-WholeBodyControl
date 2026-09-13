"""Draft ordinary70000→75000 continuation; requires real final reviewed branch inputs."""
import math
from pathlib import Path
import numpy as np
import torch
from fit_linear_head import make_actor,export
from continue_linear_fit import rng_save,rng_restore
from fit_aggregate_once import same_tree
from fit_phase_fullbatch_once import objective
from chord_common import BASE,NEW,read,write,write_atomic,sha,archive,exact,local
from chord_training_objective import strata_for,sample_pairs,chord_loss
from chord_fit_diagnostics import complete_chord_metrics,nominal_metrics
from physical_fit_diagnostics import evaluate_all,physical_metrics
from physical_training_inputs import load_physical,validate_bound_review
from physical_training_objective import physical_response_loss,expected_budgets
from chord_head_sensitivity import analyze as analyze_sensitivity
from training_runtime_identity import identity as runtime_identity

PRIOR=NEW/'velocity_chord_student_v1'
LEGACY=NEW/'fast_controller_phase_fit_v1'
GENERATION=PRIOR/'generation'
DEST=BASE/'fit'
STATE=dict(stage='INPUT_PREFLIGHT',completed_steps=70000,attempted_step=None,
    head_onnx_calls_attempted=0,head_onnx_calls_returned=0,diagnostic_torch_rows_attempted=0,diagnostic_torch_rows_returned=0,
    analytical_head_evaluations_attempted=0,analytical_head_evaluations_returned=0,
    training_head_rows_attempted=0,training_head_rows_returned=0,optimization_completed=False,
    final_export_diagnostics_completed=False,numerical_gate_passed=False,
    optimizer_call_started=False,optimizer_call_returned=False,optimizer_step_counters=[],
    committed_sample_rows=0,committed_loss_rows=0)
ACTIVE={}
STATE['legacy_diagnostics']=dict(head_onnx_calls_attempted=0,head_onnx_calls_returned=0,
    diagnostic_torch_rows_attempted=0,diagnostic_torch_rows_returned=0)
STATE['physical_diagnostics']=dict(head_onnx_calls_attempted=0,head_onnx_calls_returned=0,
    diagnostic_torch_rows_attempted=0,diagnostic_torch_rows_returned=0)

def check_inputs():
    # No model/optimizer/graph is constructed until every actual launch gate passes.
    receipt=read(BASE/'training_frozen_inputs.json')
    request=read(BASE/'training_request.json')
    assert sha(BASE/'training_request.json')==receipt['training_request_sha256']
    assert request['kind']=='one_physical_response_continuation_from70000'
    assert request['additional_updates']==5000 and request['ordinary_final_step']==75000
    assert request['coefficients']==dict(nominal=1.,velocity=1.,physical=1.)
    assert request['budgets']==expected_budgets(request['valid_rows'])
    assert 0<request['valid_rows']<=3054
    assert request['input_sha256']==receipt['input_sha256']
    for name,digest in receipt['source_sha256'].items():assert sha(Path(__file__).parent/name)==digest,name
    for name,digest in receipt['input_sha256'].items():assert sha(local(name))==digest,name
    clearance=read(BASE/'training_clearance.json')
    assert clearance['approved'] is True and clearance['model_fitting_authorized'] is True
    assert clearance['additional_updates']==5000 and clearance['ordinary_final_step']==75000
    assert clearance['frozen_receipt_sha256']==sha(BASE/'training_frozen_inputs.json')
    assert clearance['training_request_sha256']==sha(BASE/'training_request.json')
    assert sha(local(clearance['source_review_path']))==clearance['source_review_sha256']
    assert clearance['source_review_sha256']==request['reviews']['source']['sha256']
    assert set(request['reviews'])=={'prior_fit','prior_export','branch_data','source'}
    for spec in request['reviews'].values():validate_bound_review(spec,receipt['input_sha256'])
    assert request['reviews']['prior_fit']['subjects'][str(PRIOR/'fit/student_head.pt').replace('\\','/')]==sha(PRIOR/'fit/student_head.pt')
    assert request['reviews']['prior_export']['subjects'][str(PRIOR/'fit/student_head.onnx').replace('\\','/')]==sha(PRIOR/'fit/student_head.onnx')
    assert request['reviews']['branch_data']['subjects'][request['physical_paths']['manifest']]==sha(local(request['physical_paths']['manifest']))
    report=read(PRIOR/'fit/report.json')
    assert report['ordinary_final_step']==70000 and report['numerical_gate_passed'] is True
    assert report['checkpoint_sha256']==sha(PRIOR/'fit/student_head.pt')
    assert report['onnx_sha256']==sha(PRIOR/'fit/student_head.onnx')
    assert sha(PRIOR/'fit/student_head.pt')=='17b8e240e3ca711e993a5c84c0221326d365f7ed892c8166bb594d804726bc3e'
    assert sha(PRIOR/'fit/student_head.onnx')=='219b86cc4decd51671cb7aa036a944741cebed684accd91f7a03de37a7694bb5'
    generated=read(GENERATION/'report.json')
    assert generated['complete'] is True and generated['all_center_byte_parity'] is True
    assert generated['inference_calls']==dict(actor=143679,backward=3057)
    assert runtime_identity()==read(PRIOR/'training_runtime_identity.json')
    return receipt,clearance,request

def main():
    receipt,clearance,training_request=check_inputs()
    initial_gate_hashes={name:sha(BASE/name) for name in ('training_request.json','training_frozen_inputs.json','training_clearance.json')}
    assert not DEST.exists()
    centers=archive(GENERATION/'centers.npz')
    physical=load_physical(training_request,centers)
    budget=physical['budgets']
    px=np.load(GENERATION/'features.npy',mmap_mode='r')
    pb=np.load(GENERATION/'base_target.npy',mmap_mode='r')
    pt=np.load(GENERATION/'teacher_target.npy',mmap_mode='r')
    assert px.shape==(3057,23,2,1069) and pb.shape==pt.shape==(3057,23,2,23)
    torch.set_num_threads(1)
    STATE['stage']='RESTORING_70000'
    saved=torch.load(PRIOR/'fit/student_head.pt',map_location='cpu',weights_only=True)
    assert saved['completed_steps']==70000
    actor=make_actor();optimizer=torch.optim.AdamW(actor.parameters(),lr=3e-6,weight_decay=1e-5)
    actor.load_state_dict(saved['actor_state']);optimizer.load_state_dict(saved['optimizer_state']);rng_restore(saved['rng'])
    assert same_tree(actor.state_dict(),saved['actor_state']) and same_tree(optimizer.state_dict(),saved['optimizer_state'])
    assert same_tree(rng_save(),saved['rng'])
    assert len(optimizer.state)==6 and all(int(s['step'])==70000 for s in optimizer.state.values())
    STATE['optimizer_step_counters']=[int(s['step']) for s in optimizer.state.values()]
    assert all(group['lr']==3e-7 and group['weight_decay']==1e-5 for group in optimizer.param_groups)
    span,mean,std=[saved[key].numpy().copy() for key in ('joint_span','feature_mean','feature_std')]
    previous_fit=archive(LEGACY/'fit/teacher_fit.npz')
    previous_prediction=archive(PRIOR/'fit/final_predictions.npz')
    exact(span,centers['joint_span'],'old span')
    exact(mean,previous_fit['feature_mean'],'old mean');exact(std,previous_fit['feature_std'],'old std')
    features=torch.from_numpy((centers['features']-mean)/std)
    labels=torch.from_numpy((centers['residual_rad']/span).astype(np.float32))
    span_t=torch.from_numpy(span)
    base_t=torch.from_numpy(centers['base_target']);target_t=torch.from_numpy(centers['expert_target'])
    strata=strata_for(centers['control'],centers['dataset'])
    assert [[len(ids) for ids in group] for group in strata]==[[100,819,100]]*3
    physical_features=torch.from_numpy((physical['features']-mean)/std)
    physical_base=torch.from_numpy(physical['base']);physical_target=torch.from_numpy(physical['target'])
    physical_successors=torch.from_numpy(physical['successor'])
    assert torch.isfinite(physical_features).all()
    DEST.mkdir(exist_ok=False)
    np.savez_compressed(DEST/'physical_row_selection.npz',requested_valid_mask=physical['valid_mask'],
        selected_rows=physical['rows'],successor_center=physical['successor'])
    write(DEST/'physical_coverage.json',physical['coverage'])
    STATE['stage']='INITIAL_EXPORT'
    export(actor,mean,std,span,DEST/'restored70000.onnx')
    assert sha(DEST/'restored70000.onnx')==sha(PRIOR/'fit/student_head.onnx')
    actual_trace=archive(LEGACY/'nominal/trace.npz')
    actual=actual_trace['features'][[250,251,252,255,260,270,278]]
    contract=read(local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))
    STATE['stage']='INITIAL_NUMERICAL_DIAGNOSTICS'
    before=evaluate_all(actor,centers['features'],px,actual,mean,std,span,DEST/'restored70000.onnx',STATE,DEST,'initial',physical)
    assert before['export_parity_passed'] and same_tree(rng_save(),saved['rng'])
    exact(before['predicted_delta'][:3057],previous_prediction['predicted_delta'][:3057],'all restored nominal predictions')
    restored_loss=float(objective(torch.from_numpy(before['normalized_centers']),labels,strata))
    assert restored_loss==read(PRIOR/'fit/report.json')['final_nominal_objective']
    pairs=read(NEW/'bfm_entry250_labels_v1/compatibility/fixed_pairs.json')
    before_nominal=nominal_metrics(before['normalized_centers'],before['predicted_delta'][:3057],centers,strata,span,pairs,70000)
    initial_chords=complete_chord_metrics(before['predicted_delta'],centers,pb,pt,strata,span)
    write(DEST/'initial_chord_metrics.json',initial_chords)
    assert initial_chords['full_nine_cell_chord_objective']==read(PRIOR/'fit/report.json')['final_chord_objective']
    exact(before['predicted_delta'],previous_prediction['predicted_delta'],'all restored nominal and velocity predictions')
    exact(before['actual_seven_delta'],previous_prediction['actual_seven_delta'],'old seven actual predictions')
    initial_physical=physical_metrics(before['predicted_delta'][:3057],before['physical']['predicted_delta'],centers,physical,span)
    initial_physical['restored_Windows_ORT_vs_collector_WSL_batch1_max_delta']=float(np.max(np.abs(
        before['physical']['onnx_predicted_delta']-physical['collector_head_delta'])))
    initial_physical['cross_platform_comparison_is_not_byte_parity']=True
    write(DEST/'initial_physical_metrics.json',initial_physical)
    write(DEST/'initial_nominal_metrics.json',before_nominal)
    initial_sensitivity,initial_jacobians=analyze_sensitivity(actor,mean,std,span,actual_trace,contract,before['actual_seven_delta'],STATE)
    write(DEST/'initial_sensitivity.json',initial_sensitivity)
    np.savez_compressed(DEST/'initial_jacobians.npz',**initial_jacobians)
    write(DEST/'restoration70000_parity.json',dict(full_model_AdamW_RNG_exact=True,
        all3057_nominal_predictions_and_loss_exact=True,old_normalization_span_exact=True,
        restored_ONNX_bytes_exact=True,head_ONNX_calls=before['onnx_calls'],head_parity=before['onnx_max_difference'],
        initial_nominal_objective=restored_loss,initial_chord_objective=initial_chords['full_nine_cell_chord_objective']))
    request=dict(first_step=70001,last_step=75000,updates=5000,lambda_chord=1.,lambda_physical=1.,full_nominal_rows=3057,
        valid_physical_rows=len(physical['rows']),requested_physical_rows=3054,physical_coverage=physical['coverage'],
        physical_pairing='same dataset nominal successor c+1; live center; unclipped total-target response',
        physical_requested_denominators=[99,819,100]*3,training_request_sha256=sha(BASE/'training_request.json'),
        input_request=training_request,budgets=budget,learning_rate_reset_after_exact_restoration=True,
        runtime=read(PRIOR/'training_runtime_identity.json'),runtime_identity_sha256=sha(PRIOR/'training_runtime_identity.json'),
        pairs_per_cell=64,cells=9,perturbed_rows_per_update=1152,training_head_rows=budget['training_head_rows'],
        sampler='dataset then phase; one torch.randint draw64 over each row×23 axis Cartesian product; global restored RNG',
        learning_rate='3e-7+0.5*(3e-6-3e-7)*(1+cos(pi*k/4999)), k=0..4999',
        prior_sha256=sha(PRIOR/'fit/student_head.pt'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        clearance_sha256=sha(BASE/'training_clearance.json'),ordinary_final_only=True,
        nominal_loss_arithmetic='original normalized residual objective retained',
        center_predictions_reused_with_gradient=True,intermediate_physics=False,hardware_authorized=False)
    write(DEST/'request.json',request)
    picked_rows=np.lib.format.open_memmap(DEST/'sampled_center_rows.npy',mode='w+',dtype=np.int32,shape=(5000,576))
    picked_axes=np.lib.format.open_memmap(DEST/'sampled_axes.npy',mode='w+',dtype=np.int8,shape=(5000,576))
    losses=[];cell_losses=[];completed=70000
    ACTIVE.update(nominal_raw_features=centers['features'],physical_raw_features=physical['features'],
        physical_valid_rows=physical['rows'],physical_successor_rows=physical['successor'])
    STATE['stage']='FIXED_5000_UPDATES'
    try:
        for k,step in enumerate(range(70001,75001)):
            STATE['attempted_step']=step
            for key in ('nominal_normalized_returned','velocity_normalized_returned','physical_normalized_returned','physical_cell_losses'):
                ACTIVE.pop(key,None)
            STATE['training_forward_stage']='nominal'
            STATE['optimizer_call_started']=False;STATE['optimizer_call_returned']=False
            rate=3e-7+.5*(3e-6-3e-7)*(1+math.cos(math.pi*k/4999))
            for group in optimizer.param_groups:group['lr']=rate
            picked,axes=sample_pairs(strata)
            ids=picked.numpy();axis=axes.numpy()
            ACTIVE.update(sampled_center_rows=ids.copy(),sampled_axes=axis.copy(),attempted_step=np.int64(step))
            selected_x=px[ids,axis].reshape(1152,1069)
            ACTIVE['velocity_raw_features']=selected_x
            STATE['training_head_rows_attempted']+=3057
            normalized_centers=actor(features);STATE['training_head_rows_returned']+=3057
            ACTIVE['nominal_normalized_returned']=normalized_centers.detach().numpy().copy()
            STATE['training_forward_stage']='velocity'
            STATE['training_head_rows_attempted']+=1152
            normalized_probes=actor(torch.from_numpy((selected_x-mean)/std));STATE['training_head_rows_returned']+=1152
            ACTIVE['velocity_normalized_returned']=normalized_probes.detach().numpy().copy()
            nominal=objective(normalized_centers,labels,strata)
            chords=chord_loss(normalized_centers,normalized_probes,picked,base_t,
                torch.from_numpy(np.asarray(pb[ids,axis])),target_t,torch.from_numpy(np.asarray(pt[ids,axis])),span_t)
            STATE['training_forward_stage']='physical'
            STATE['training_head_rows_attempted']+=len(physical['rows'])
            normalized_physical=actor(physical_features);STATE['training_head_rows_returned']+=len(physical['rows'])
            ACTIVE['physical_normalized_returned']=normalized_physical.detach().numpy().copy()
            physical_loss,physical_cell_losses=physical_response_loss(normalized_centers,normalized_physical,
                physical_successors,base_t,physical_base,target_t,physical_target,span_t,physical['cells'])
            ACTIVE['physical_cell_losses']=physical_cell_losses.detach().numpy().copy()
            loss=nominal+chords+physical_loss
            # Reporting from returned tensors only; original loss graph remains unchanged.
            with torch.no_grad():
                nominal_square=(normalized_centers-labels)**2
                nominal_cells=torch.stack([nominal_square[ix].mean() for group in strata for ix in group])
                center_proposal=base_t[picked]+normalized_centers[picked]*span_t
                velocity_proposal=torch.from_numpy(np.asarray(pb[ids,axis]))+(normalized_probes*span_t).reshape(576,2,23)
                velocity_change=torch.from_numpy(np.asarray(pt[ids,axis]))-target_t[picked,None]
                velocity_square=(((velocity_proposal-center_proposal[:,None])-velocity_change)/span_t)**2
                velocity_cells=torch.stack([velocity_square[cell*64:(cell+1)*64].mean() for cell in range(9)])
                current_cells=np.stack([nominal_cells.numpy(),velocity_cells.numpy(),physical_cell_losses.detach().numpy()])
            ACTIVE['current_cell_losses']=current_cells.copy()
            if not torch.isfinite(loss):raise ValueError('Nonfinite combined objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),10.,error_if_nonfinite=True)
            STATE['optimizer_call_started']=True
            optimizer.step()
            STATE['optimizer_call_returned']=True
            STATE['optimizer_step_counters']=[int(s['step']) for s in optimizer.state.values()]
            completed=step
            STATE['completed_steps']=completed
            assert all(torch.isfinite(p).all() for p in actor.parameters())
            assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
            picked_rows[k]=ids;picked_axes[k]=axis
            losses.append([float(nominal.detach()),float(chords.detach()),float(physical_loss.detach()),float(loss.detach()),rate])
            cell_losses.append(current_cells.copy())
            STATE['committed_sample_rows']=k+1;STATE['committed_loss_rows']=len(losses)
            if (k+1)%50==0:
                picked_rows.flush();picked_axes.flush()
                np.save(DEST/'training_progress.npy',np.asarray(losses))
                np.save(DEST/'training_cell_losses.npy',np.asarray(cell_losses))
                write_atomic(DEST/'progress.json',dict(completed_steps=completed,additional_updates=k+1,
                    nominal_objective=losses[-1][0],sampled_chord_objective=losses[-1][1],physical_objective=losses[-1][2],combined_objective=losses[-1][3],learning_rate=rate))
                print('step',step,'nominal',losses[-1][0],'chord',losses[-1][1],flush=True)
    except BaseException as exc:
        picked_rows.flush();picked_axes.flush()
        STATE['optimizer_step_counters']=[int(s['step']) for s in optimizer.state.values()]
        STATE['committed_loss_rows']=len(losses)
        np.save(DEST/'training_progress.npy',np.asarray(losses,dtype=np.float64).reshape(-1,5))
        np.save(DEST/'training_cell_losses.npy',np.asarray(cell_losses,dtype=np.float64).reshape(-1,3,9))
        torch.save(dict(actor_state=actor.state_dict(),optimizer_state=optimizer.state_dict(),rng=rng_save(),
            feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),joint_span=span_t,
            completed_steps=completed,request=request,attempt_state=STATE.copy(),
            automatic_resume_qualified=False),DEST/'failed_progress.pt')
        write(DEST/'fit_failure.json',dict(completed_steps=completed,attempted_step=step,
            exception=type(exc).__name__,message=str(exc),failed_checkpoint_sha256=sha(DEST/'failed_progress.pt'),
            attempt_state=STATE.copy(),automatic_resume_qualified=False,
            failed_optimizer_call_may_be_partial=bool(STATE['optimizer_call_started'] and not STATE['optimizer_call_returned'])))
        raise
    picked_rows.flush();picked_axes.flush()
    assert completed==75000 and len(losses)==5000
    assert all(int(s['step'])==75000 for s in optimizer.state.values())
    assert STATE['training_head_rows_attempted']==STATE['training_head_rows_returned']==budget['training_head_rows']
    STATE['optimization_completed']=True;STATE['stage']='SAVING_ORDINARY_FINAL'
    checkpoint=dict(kind=saved['kind'],actor_state=actor.state_dict(),optimizer_state=optimizer.state_dict(),rng=rng_save(),
        feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),joint_span=span_t,
        completed_steps=75000,additional_updates=5000,request=request)
    torch.save(checkpoint,DEST/'student_head.pt')
    np.savez_compressed(DEST/'training_arrays.npz',global_step=np.arange(70001,75001),
        nominal_objective=np.asarray(losses)[:,0],sampled_chord_objective=np.asarray(losses)[:,1],
        cell_losses=np.asarray(cell_losses),
        physical_objective=np.asarray(losses)[:,2],training_objective=np.asarray(losses)[:,3],learning_rate=np.asarray(losses)[:,4])
    write(DEST/'optimization_completed.json',dict(ordinary_final_step=75000,checkpoint_sha256=sha(DEST/'student_head.pt'),
        training_arrays_sha256=sha(DEST/'training_arrays.npz'),export_validation_pending=True))
    STATE['stage']='FINAL_EXPORT';export(actor,mean,std,span,DEST/'student_head.onnx')
    STATE['stage']='FINAL_NUMERICAL_DIAGNOSTICS'
    final=evaluate_all(actor,centers['features'],px,actual,mean,std,span,DEST/'student_head.onnx',STATE,DEST,'final',physical)
    final_chords=complete_chord_metrics(final['predicted_delta'],centers,pb,pt,strata,span)
    final_nominal=nominal_metrics(final['normalized_centers'],final['predicted_delta'][:3057],centers,strata,span,pairs,75000)
    write(DEST/'final_chord_metrics.json',final_chords)
    write(DEST/'final_nominal_metrics.json',final_nominal)
    final_physical=physical_metrics(final['predicted_delta'][:3057],final['physical']['predicted_delta'],centers,physical,span)
    write(DEST/'final_physical_metrics.json',final_physical)
    final_sensitivity,final_jacobians=analyze_sensitivity(actor,mean,std,span,actual_trace,contract,final['actual_seven_delta'],STATE)
    write(DEST/'final_sensitivity.json',final_sensitivity)
    np.savez_compressed(DEST/'final_jacobians.npz',**final_jacobians)
    STATE['final_export_diagnostics_completed']=True;STATE['numerical_gate_passed']=bool(final['export_parity_passed'])
    report=dict(completed=bool(final['export_parity_passed']),optimization_completed=True,
        final_export_diagnostics_completed=True,numerical_gate_passed=bool(final['export_parity_passed']),ordinary_final_step=75000,additional_updates=5000,
        head_ONNX_calls=before['onnx_calls']+final['onnx_calls'],final_head_parity=final['onnx_max_difference'],
        export_parity_passed=final['export_parity_passed'],initial_nominal_objective=restored_loss,
        final_nominal_objective=final_nominal['nine_cell_nominal_objective'],
        initial_chord_objective=initial_chords['full_nine_cell_chord_objective'],
        final_chord_objective=final_chords['full_nine_cell_chord_objective'],
        initial_physical_objective=initial_physical['nine_cell_response_objective'],
        final_physical_objective=final_physical['nine_cell_response_objective'],
        physical_coverage=physical['coverage'],physical_data_manifest_sha256=sha(local(training_request['physical_paths']['manifest'])),
        checkpoint_sha256=sha(DEST/'student_head.pt'),onnx_sha256=sha(DEST/'student_head.onnx'),
        physical_evaluations=0,new_expert_queries=0,checkpoint_selection=False,hardware_authorized=False)
    assert report['head_ONNX_calls']==budget['head_onnx_calls']
    assert STATE['diagnostic_torch_rows_attempted']==STATE['diagnostic_torch_rows_returned']==budget['diagnostic_torch_rows']
    assert STATE['analytical_head_evaluations_attempted']==STATE['analytical_head_evaluations_returned']==14
    assert STATE['legacy_diagnostics']['head_onnx_calls_attempted']==STATE['legacy_diagnostics']['head_onnx_calls_returned']==1126
    assert STATE['head_onnx_calls_attempted']==STATE['head_onnx_calls_returned']==budget['head_onnx_calls']
    STATE['stage']='FINAL_FROZEN_INPUT_REHASH'
    assert all(sha(BASE/name)==digest for name,digest in initial_gate_hashes.items())
    check_inputs()
    report['attempt_counters']=STATE.copy()
    write(DEST/'report.json',report)
    assert final['export_parity_passed'],'Final ONNX parity failed; preserved ordinary final, no rollout'
    STATE['stage']='COMPLETE';write_atomic(DEST/'attempt_status.json',STATE)

def guarded_main():
    try:
        main()
    except BaseException as exc:
        STATE['failure']=dict(type=type(exc).__name__,message=str(exc))
        STATE['failed_stage']=STATE['stage'];STATE['stage']='FAILED'
        if DEST.exists():
            if ACTIVE:np.savez_compressed(DEST/'failed_attempt.npz',**ACTIVE)
            STATE['preserved_checkpoints']={p.name:sha(p) for p in DEST.glob('*.pt')}
            write_atomic(DEST/'attempt_status.json',STATE)
        else:
            write_atomic(BASE/'fit_preflight_failure.json',STATE)
        raise

if __name__=='__main__':guarded_main()
