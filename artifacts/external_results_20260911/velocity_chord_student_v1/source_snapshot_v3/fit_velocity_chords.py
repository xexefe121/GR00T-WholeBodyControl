"""One fixed ordinary65000→70000 continuation, gated on reviewed generated data."""
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
from chord_fit_diagnostics import evaluate_fixed,complete_chord_metrics,nominal_metrics
from chord_head_sensitivity import analyze as analyze_sensitivity
from training_runtime_identity import identity as runtime_identity

PRIOR=NEW/'fast_controller_phase_fit_v1'
DEST=BASE/'fit'
STATE=dict(stage='INPUT_PREFLIGHT',completed_steps=65000,attempted_step=None,
    head_onnx_calls_attempted=0,head_onnx_calls_returned=0,diagnostic_torch_rows_attempted=0,diagnostic_torch_rows_returned=0,
    analytical_head_evaluations_attempted=0,analytical_head_evaluations_returned=0,
    training_head_rows_attempted=0,training_head_rows_returned=0,optimization_completed=False,
    final_export_diagnostics_completed=False,numerical_gate_passed=False,
    optimizer_call_started=False,optimizer_call_returned=False,optimizer_step_counters=[],
    committed_sample_rows=0,committed_loss_rows=0)
ACTIVE={}

def check_inputs():
    receipt=read(BASE/'training_frozen_inputs.json')
    for name,digest in receipt['source_sha256'].items():assert sha(Path(__file__).parent/name)==digest,name
    for name,digest in receipt['input_sha256'].items():assert sha(local(name))==digest,name
    clearance=read(BASE/'training_clearance.json')
    assert clearance['approved'] is True and clearance['additional_updates']==5000
    assert clearance['ordinary_final_step']==70000
    assert clearance['frozen_receipt_sha256']==sha(BASE/'training_frozen_inputs.json')
    assert sha(local(clearance['dataset_review_path']))==clearance['dataset_review_sha256']
    assert sha(local(clearance['source_review_path']))==clearance['source_review_sha256']
    report=read(BASE/'generation/report.json')
    assert report['complete'] is True and report['all_center_byte_parity'] is True
    assert report['inference_calls']==dict(actor=143679,backward=3057)
    assert runtime_identity()==read(BASE/'training_runtime_identity.json')
    return receipt,clearance

def main():
    receipt,clearance=check_inputs()
    assert not DEST.exists()
    centers=archive(BASE/'generation/centers.npz')
    px=np.load(BASE/'generation/features.npy',mmap_mode='r')
    pb=np.load(BASE/'generation/base_target.npy',mmap_mode='r')
    pt=np.load(BASE/'generation/teacher_target.npy',mmap_mode='r')
    assert px.shape==(3057,23,2,1069) and pb.shape==pt.shape==(3057,23,2,23)
    torch.set_num_threads(1)
    STATE['stage']='RESTORING_65000'
    saved=torch.load(PRIOR/'fit/student_head.pt',map_location='cpu',weights_only=True)
    assert saved['completed_steps']==65000
    actor=make_actor();optimizer=torch.optim.AdamW(actor.parameters(),lr=3e-6,weight_decay=1e-5)
    actor.load_state_dict(saved['actor_state']);optimizer.load_state_dict(saved['optimizer_state']);rng_restore(saved['rng'])
    assert same_tree(actor.state_dict(),saved['actor_state']) and same_tree(optimizer.state_dict(),saved['optimizer_state'])
    assert same_tree(rng_save(),saved['rng'])
    assert len(optimizer.state)==6 and all(int(s['step'])==65000 for s in optimizer.state.values())
    STATE['optimizer_step_counters']=[int(s['step']) for s in optimizer.state.values()]
    assert all(group['lr']==3e-6 and group['weight_decay']==1e-5 for group in optimizer.param_groups)
    span,mean,std=[saved[key].numpy().copy() for key in ('joint_span','feature_mean','feature_std')]
    previous_fit=archive(PRIOR/'fit/teacher_fit.npz')
    exact(span,centers['joint_span'],'old span')
    exact(mean,previous_fit['feature_mean'],'old mean');exact(std,previous_fit['feature_std'],'old std')
    features=torch.from_numpy((centers['features']-mean)/std)
    labels=torch.from_numpy((centers['residual_rad']/span).astype(np.float32))
    span_t=torch.from_numpy(span)
    base_t=torch.from_numpy(centers['base_target']);target_t=torch.from_numpy(centers['expert_target'])
    strata=strata_for(centers['control'],centers['dataset'])
    assert [[len(ids) for ids in group] for group in strata]==[[100,819,100]]*3
    DEST.mkdir(exist_ok=False)
    STATE['stage']='INITIAL_EXPORT'
    export(actor,mean,std,span,DEST/'restored65000.onnx')
    assert sha(DEST/'restored65000.onnx')==sha(PRIOR/'fit/student_head.onnx')
    actual_trace=archive(PRIOR/'nominal/trace.npz')
    actual=actual_trace['features'][[250,251,252,255,260,270,278]]
    contract=read(local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))
    STATE['stage']='INITIAL_NUMERICAL_DIAGNOSTICS'
    before=evaluate_fixed(actor,centers['features'],px,actual,mean,std,span,DEST/'restored65000.onnx',STATE,DEST,'initial')
    assert before['export_parity_passed'] and same_tree(rng_save(),saved['rng'])
    exact(before['predicted_delta'][:3057],previous_fit['predicted_delta'],'all restored nominal predictions')
    restored_loss=float(objective(torch.from_numpy(before['normalized_centers']),labels,strata))
    assert restored_loss==read(PRIOR/'fit/report.json')['final_nine_cell_objective']
    pairs=read(NEW/'bfm_entry250_labels_v1/compatibility/fixed_pairs.json')
    before_nominal=nominal_metrics(before['normalized_centers'],before['predicted_delta'][:3057],centers,strata,span,pairs,65000)
    initial_chords=complete_chord_metrics(before['predicted_delta'],centers,pb,pt,strata,span)
    write(DEST/'initial_chord_metrics.json',initial_chords)
    write(DEST/'initial_nominal_metrics.json',before_nominal)
    initial_sensitivity,initial_jacobians=analyze_sensitivity(actor,mean,std,span,actual_trace,contract,before['actual_seven_delta'],STATE)
    write(DEST/'initial_sensitivity.json',initial_sensitivity)
    np.savez_compressed(DEST/'initial_jacobians.npz',**initial_jacobians)
    write(DEST/'restoration65000_parity.json',dict(full_model_AdamW_RNG_exact=True,
        all3057_nominal_predictions_and_loss_exact=True,old_normalization_span_exact=True,
        restored_ONNX_bytes_exact=True,head_ONNX_calls=before['onnx_calls'],head_parity=before['onnx_max_difference'],
        initial_nominal_objective=restored_loss,initial_chord_objective=initial_chords['full_nine_cell_chord_objective']))
    request=dict(first_step=65001,last_step=70000,updates=5000,lambda_chord=1.,full_nominal_rows=3057,
        runtime=read(BASE/'training_runtime_identity.json'),runtime_identity_sha256=sha(BASE/'training_runtime_identity.json'),
        pairs_per_cell=64,cells=9,perturbed_rows_per_update=1152,training_head_rows=21045000,
        sampler='dataset then phase; one torch.randint draw64 over each row×23 axis Cartesian product; global restored RNG',
        learning_rate='3e-7+0.5*(3e-6-3e-7)*(1+cos(pi*k/4999)), k=0..4999',
        prior_sha256=sha(PRIOR/'fit/student_head.pt'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        clearance_sha256=sha(BASE/'training_clearance.json'),ordinary_final_only=True,
        nominal_loss_arithmetic='original normalized residual objective retained',
        center_predictions_reused_with_gradient=True,intermediate_physics=False,hardware_authorized=False)
    write(DEST/'request.json',request)
    picked_rows=np.lib.format.open_memmap(DEST/'sampled_center_rows.npy',mode='w+',dtype=np.int32,shape=(5000,576))
    picked_axes=np.lib.format.open_memmap(DEST/'sampled_axes.npy',mode='w+',dtype=np.int8,shape=(5000,576))
    losses=[];completed=65000
    STATE['stage']='FIXED_5000_UPDATES'
    try:
        for k,step in enumerate(range(65001,70001)):
            STATE['attempted_step']=step
            STATE['optimizer_call_started']=False;STATE['optimizer_call_returned']=False
            rate=3e-7+.5*(3e-6-3e-7)*(1+math.cos(math.pi*k/4999))
            for group in optimizer.param_groups:group['lr']=rate
            picked,axes=sample_pairs(strata)
            ids=picked.numpy();axis=axes.numpy()
            ACTIVE.update(sampled_center_rows=ids.copy(),sampled_axes=axis.copy(),attempted_step=np.int64(step))
            selected_x=px[ids,axis].reshape(1152,1069)
            STATE['training_head_rows_attempted']+=3057
            normalized_centers=actor(features);STATE['training_head_rows_returned']+=3057
            STATE['training_head_rows_attempted']+=1152
            normalized_probes=actor(torch.from_numpy((selected_x-mean)/std));STATE['training_head_rows_returned']+=1152
            nominal=objective(normalized_centers,labels,strata)
            chords=chord_loss(normalized_centers,normalized_probes,picked,base_t,
                torch.from_numpy(np.asarray(pb[ids,axis])),target_t,torch.from_numpy(np.asarray(pt[ids,axis])),span_t)
            loss=nominal+chords
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
            losses.append([float(nominal.detach()),float(chords.detach()),float(loss.detach()),rate])
            STATE['committed_sample_rows']=k+1;STATE['committed_loss_rows']=len(losses)
            if (k+1)%50==0:
                picked_rows.flush();picked_axes.flush()
                np.save(DEST/'training_progress.npy',np.asarray(losses))
                write_atomic(DEST/'progress.json',dict(completed_steps=completed,additional_updates=k+1,
                    nominal_objective=losses[-1][0],sampled_chord_objective=losses[-1][1],combined_objective=losses[-1][2],learning_rate=rate))
                print('step',step,'nominal',losses[-1][0],'chord',losses[-1][1],flush=True)
    except BaseException as exc:
        picked_rows.flush();picked_axes.flush()
        STATE['optimizer_step_counters']=[int(s['step']) for s in optimizer.state.values()]
        STATE['committed_loss_rows']=len(losses)
        np.save(DEST/'training_progress.npy',np.asarray(losses,dtype=np.float64).reshape(-1,4))
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
    assert completed==70000 and len(losses)==5000
    assert all(int(s['step'])==70000 for s in optimizer.state.values())
    assert STATE['training_head_rows_attempted']==STATE['training_head_rows_returned']==21045000
    STATE['optimization_completed']=True;STATE['stage']='SAVING_ORDINARY_FINAL'
    checkpoint=dict(kind=saved['kind'],actor_state=actor.state_dict(),optimizer_state=optimizer.state_dict(),rng=rng_save(),
        feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),joint_span=span_t,
        completed_steps=70000,additional_updates=5000,request=request)
    torch.save(checkpoint,DEST/'student_head.pt')
    np.savez_compressed(DEST/'training_arrays.npz',global_step=np.arange(65001,70001),
        nominal_objective=np.asarray(losses)[:,0],sampled_chord_objective=np.asarray(losses)[:,1],
        training_objective=np.asarray(losses)[:,2],learning_rate=np.asarray(losses)[:,3])
    write(DEST/'optimization_completed.json',dict(ordinary_final_step=70000,checkpoint_sha256=sha(DEST/'student_head.pt'),
        training_arrays_sha256=sha(DEST/'training_arrays.npz'),export_validation_pending=True))
    STATE['stage']='FINAL_EXPORT';export(actor,mean,std,span,DEST/'student_head.onnx')
    STATE['stage']='FINAL_NUMERICAL_DIAGNOSTICS'
    final=evaluate_fixed(actor,centers['features'],px,actual,mean,std,span,DEST/'student_head.onnx',STATE,DEST,'final')
    final_chords=complete_chord_metrics(final['predicted_delta'],centers,pb,pt,strata,span)
    final_nominal=nominal_metrics(final['normalized_centers'],final['predicted_delta'][:3057],centers,strata,span,pairs,70000)
    write(DEST/'final_chord_metrics.json',final_chords)
    write(DEST/'final_nominal_metrics.json',final_nominal)
    final_sensitivity,final_jacobians=analyze_sensitivity(actor,mean,std,span,actual_trace,contract,final['actual_seven_delta'],STATE)
    write(DEST/'final_sensitivity.json',final_sensitivity)
    np.savez_compressed(DEST/'final_jacobians.npz',**final_jacobians)
    STATE['final_export_diagnostics_completed']=True;STATE['numerical_gate_passed']=bool(final['export_parity_passed'])
    report=dict(completed=bool(final['export_parity_passed']),optimization_completed=True,
        final_export_diagnostics_completed=True,numerical_gate_passed=bool(final['export_parity_passed']),ordinary_final_step=70000,additional_updates=5000,
        head_ONNX_calls=before['onnx_calls']+final['onnx_calls'],final_head_parity=final['onnx_max_difference'],
        export_parity_passed=final['export_parity_passed'],initial_nominal_objective=restored_loss,
        final_nominal_objective=final_nominal['nine_cell_nominal_objective'],
        initial_chord_objective=initial_chords['full_nine_cell_chord_objective'],
        final_chord_objective=final_chords['full_nine_cell_chord_objective'],
        checkpoint_sha256=sha(DEST/'student_head.pt'),onnx_sha256=sha(DEST/'student_head.onnx'),
        physical_evaluations=0,new_expert_queries=0,checkpoint_selection=False,hardware_authorized=False)
    assert report['head_ONNX_calls']==1126
    assert STATE['diagnostic_torch_rows_attempted']==STATE['diagnostic_torch_rows_returned']==287372
    assert STATE['analytical_head_evaluations_attempted']==STATE['analytical_head_evaluations_returned']==14
    STATE['stage']='FINAL_FROZEN_INPUT_REHASH'
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
