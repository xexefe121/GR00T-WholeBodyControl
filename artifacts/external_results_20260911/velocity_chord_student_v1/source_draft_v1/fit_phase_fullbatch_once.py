"""One conditional5000-update full-batch phase fit; ordinaryfinal65000 only."""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import onnxruntime as ort
from fit_linear_head import CONFIG,make_actor,export
from continue_linear_fit import rng_save,rng_restore
from fit_aggregate_once import same_tree
from student_linear_runtime import BASE,KIND,FROZEN_RECEIPT,assert_frozen,archive,sha,finite_json

PRIOR=BASE.parent/'fast_controller_aggregate_fit_v1'
LABEL_ROOT=BASE.parent/'bfm_entry250_labels_v1'
PATHS=(BASE.parent/'fast_controller_nominal_pilot_v1/labels/labels.npz',
       BASE.parent/'fresh_expert_labels_resume_v1/labels/labels.npz',LABEL_ROOT/'labels/labels.npz')
NAMES=('old','query1','query250');PHASES=('acquisition','source','return')
DEST=BASE/'fit';FIRST,LAST=60001,65000

def write(name,value):
    (DEST/name).write_text(json.dumps(finite_json(value),indent=2,allow_nan=False)+'\n')

def phase_indices(control):
    return [np.flatnonzero(mask) for mask in ((control>=250)&(control<350),
        (control>=350)&(control<1169),(control>=1169)&(control<1269))]

def objective(prediction,labels,strata):
    square=(prediction-labels)**2
    return torch.stack([square[ids].mean() for group in strata for ids in group]).mean()

def summarize(error,residual_error,indices):
    e,r=error[indices],residual_error[indices]
    return dict(rows=len(e),applied_target_rmse_rad=float(np.sqrt(np.mean(e**2))),
        applied_target_rmse_by_joint_rad=np.sqrt(np.mean(e**2,axis=0)).tolist(),
        applied_target_abs_error_p95_by_joint_rad=np.percentile(np.abs(e),95,axis=0).tolist(),
        applied_target_abs_error_max_by_joint_rad=np.max(np.abs(e),axis=0).tolist(),
        preclip_residual_rmse_rad=float(np.sqrt(np.mean(r**2))))

def diagnostics(actor,features,labels,span,data,strata,pairs,step):
    with torch.inference_mode():
        normalized=actor(features)
        loss=float(objective(normalized,labels,strata))
        predicted=(normalized*torch.from_numpy(span)).numpy()
    targets=np.clip(data['base_target']+predicted,data['joint_limits'][:,0],data['joint_limits'][:,1])
    error=targets-data['expert_target'];residual_error=predicted-data['residual_rad']
    assert math.isfinite(loss) and np.isfinite(predicted).all() and np.isfinite(error).all()
    summary={'all':summarize(error,residual_error,np.arange(3057))}
    for code,name in enumerate(NAMES):
        indices=np.arange(code*1019,(code+1)*1019)
        summary[name]=dict(all=summarize(error,residual_error,indices),
            phases={phase:summarize(error,residual_error,ids) for phase,ids in zip(PHASES,strata[code])},
            first24_each_phase={phase:summarize(error,residual_error,ids[:24]) for phase,ids in zip(PHASES,strata[code])})
    pair_rows=[]
    for pair in pairs:
        i,j=pair['a_global_row'],pair['b_global_row']
        assert 0<=i<3057 and 0<=j<3057
        assert pair['a_control']==int(data['control'][i]) and pair['b_control']==int(data['control'][j])
        pair_rows.append(dict(pair,predicted_target_difference_rad=(targets[j]-targets[i]).tolist(),
            actual_target_difference_rad=(data['expert_target'][j]-data['expert_target'][i]).tolist(),
            a_target_error_rad=error[i].tolist(),b_target_error_rad=error[j].tolist(),
            predicted_residual_difference_rad=(predicted[j]-predicted[i]).tolist(),
            actual_residual_difference_rad=(data['residual_rad'][j]-data['residual_rad'][i]).tolist()))
    query=2038
    result=dict(global_step=step,additional_updates=step-60000,nine_cell_normalized_residual_objective=loss,
        datasets=summary,actual_query250=dict(target_error_rad=error[query].tolist(),
            residual_error_rad=residual_error[query].tolist(),target_rmse_rad=float(np.sqrt(np.mean(error[query]**2)))),
        fixed_pairs=pair_rows,fit_diagnostics_only=True,checkpoint_selection=False,physical_evaluations=0)
    return result,predicted,targets,error

def main():
    assert_frozen()
    assert not DEST.exists(),'Existing phase fit attempt must remain preserved.'
    clearance=json.loads((BASE/'training_clearance.json').read_text())
    assert clearance['model_fitting_authorized'] is True and clearance['additional_updates']==5000 and clearance['ordinary_final_global_step']==65000
    assert clearance['frozen_sources_sha256']==sha(FROZEN_RECEIPT)
    compatibility=json.loads((LABEL_ROOT/'compatibility/report.json').read_text())
    assert compatibility['compatibility_pass'] and compatibility['exact_conflict_groups']==0
    assert clearance['compatibility_report_sha256']==sha(LABEL_ROOT/'compatibility/report.json')
    assert compatibility['samples_each']==1019 and compatibility['total_samples']==3057
    all_data=[archive(p) for p in PATHS]
    for name,path in zip(NAMES,PATHS):assert compatibility['label_sha256'][name]==sha(path)
    np.testing.assert_array_equal(all_data[0]['control'],np.arange(1269))
    np.testing.assert_array_equal(all_data[1]['control'],np.arange(1,1269))
    np.testing.assert_array_equal(all_data[2]['control'],np.arange(250,1269))
    selected=[]
    for original in all_data:
        ids=np.flatnonzero((original['control']>=250)&(original['control']<1269))
        np.testing.assert_array_equal(original['control'][ids],np.arange(250,1269))
        np.testing.assert_array_equal(original['source_frame'][ids],np.arange(261,1280))
        selected.append({key:original[key][ids] for key in ('features','residual_rad','base_target','expert_target','control','source_frame')})
    data={key:np.concatenate([d[key] for d in selected]) for key in selected[0]}
    data['joint_limits']=all_data[0]['joint_limits'].copy()
    torch.set_num_threads(1)
    saved=torch.load(PRIOR/'fit/student_head.pt',map_location='cpu',weights_only=True)
    assert sha(PRIOR/'fit/student_head.pt')=='484d20d30408edccb00b22370f1ddf2152d04e4cdabbd92272d4d19175d1900b'
    assert saved['completed_steps']==60000
    span,mean,std=[saved[key].numpy().copy() for key in ('joint_span','feature_mean','feature_std')]
    previous_fit=archive(PRIOR/'fit/teacher_fit.npz')
    for key,actual in (('feature_mean',mean),('feature_std',std)):np.testing.assert_array_equal(actual,previous_fit[key])
    for d in all_data:
        np.testing.assert_array_equal(span,d['joint_span']);np.testing.assert_array_equal(data['joint_limits'],d['joint_limits'])
    X=data['features'].astype(np.float32);features=torch.from_numpy((X-mean)/std)
    labels=torch.from_numpy((data['residual_rad']/span).astype(np.float32))
    assert features.shape==(3057,1069) and torch.isfinite(features).all() and torch.isfinite(labels).all()
    datasets=np.repeat(np.arange(3),1019)
    strata=[[ids+code*1019 for ids in phase_indices(d['control'])] for code,d in enumerate(selected)]
    assert [[len(ids) for ids in group] for group in strata]==[[100,819,100]]*3
    actor=make_actor();optimizer=torch.optim.AdamW(actor.parameters(),lr=3e-4,weight_decay=1e-5)
    actor.load_state_dict(saved['actor_state']);optimizer.load_state_dict(saved['optimizer_state']);rng_restore(saved['rng'])
    assert same_tree(actor.state_dict(),saved['actor_state']) and same_tree(optimizer.state_dict(),saved['optimizer_state'])
    assert same_tree(rng_save(),saved['rng']) and len(optimizer.state)==6
    assert all(int(state['step'])==60000 for state in optimizer.state.values())
    assert all(group['lr']==3e-4 and group['weight_decay']==1e-5 for group in optimizer.param_groups)
    prior_X=np.concatenate([d['features'] for d in all_data[:2]])
    with torch.inference_mode():prior_predicted=(actor(torch.from_numpy((prior_X-mean)/std))*torch.from_numpy(span)).numpy()
    np.testing.assert_array_equal(prior_predicted,previous_fit['predicted_delta'])
    DEST.mkdir(exist_ok=False)
    export(actor,mean,std,span,DEST/'restored60000.onnx')
    assert sha(DEST/'restored60000.onnx')==sha(PRIOR/'fit/student_head.onnx')
    assert same_tree(rng_save(),saved['rng'])
    pairs=json.loads((LABEL_ROOT/'compatibility/fixed_pairs.json').read_text())
    assert sha(LABEL_ROOT/'compatibility/fixed_pairs.json')==compatibility['fixed_pairs_sha256']
    request=dict(kind=KIND,experiment='one_phase_only_fullbatch_continuation',first_global_step=FIRST,last_global_step=LAST,
        additional_updates=5000,samples_each=1019,total_samples=3057,phase_rows=[[100,819,100]]*3,
        fixed_dataset_order=list(NAMES),sampling='none; every3057 rows once per update, ascending controls within dataset',
        objective='mean of nine dataset-by-phase MSE means over rows and23 normalized residual outputs',
        learning_rate='3e-6+0.5*(3e-5-3e-6)*(1+cos(pi*k/4999)), k=0..4999',
        optimizer=dict(name='AdamW',weight_decay=1e-5,gradient_clip_norm=10.,device='cpu',threads=1),
        original_configuration=CONFIG,original_normalization_refitted=False,full_final60000_model_optimizer_rng_restored=True,
        prior_checkpoint_sha256=sha(PRIOR/'fit/student_head.pt'),label_sha256={name:sha(path) for name,path in zip(NAMES,PATHS)},
        compatibility_report_sha256=sha(LABEL_ROOT/'compatibility/report.json'),frozen_sources_sha256=sha(FROZEN_RECEIPT),
        training_clearance_sha256=sha(BASE/'training_clearance.json'),ordinary_final_only=True,early_stopping=False,
        diagnostics_every_updates=250,intermediate_physics_evaluations=False,new_expert_queries=False,hardware_authorized=False)
    write('request.json',request)
    write('restoration60000_parity.json',dict(model_tensors_exact=True,optimizer_tree_exact=True,rng_exact=True,
        all2537_prior_predictions_exact=True,normalization_span_exact=True,restored_ONNX_bytes_exact=True,
        optimizer_parameter_states=6,optimizer_steps=60000,updates_executed=0,prior_checkpoint_sha256=request['prior_checkpoint_sha256']))
    write('fixed_pairs.json',pairs)
    before,_,_,_=diagnostics(actor,features,labels,span,data,strata,pairs,60000)
    history=[before];losses=[];rates=[];completed=60000;started=time.perf_counter()
    assert same_tree(rng_save(),saved['rng'])
    try:
        for k,step in enumerate(range(FIRST,LAST+1)):
            rate=3e-6+.5*(3e-5-3e-6)*(1+math.cos(math.pi*k/4999))
            for group in optimizer.param_groups:group['lr']=rate
            predicted=actor(features);loss=objective(predicted,labels,strata)
            if not torch.isfinite(loss):raise ValueError('nonfinite phase training loss at step%d'%step)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),10.,error_if_nonfinite=True)
            optimizer.step();completed=step
            if not all(torch.isfinite(p).all() for p in actor.parameters()):raise ValueError('nonfinite phase model state')
            if not all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)):
                raise ValueError('nonfinite phase optimizer state')
            losses.append(float(loss.detach()));rates.append(rate)
            if (step-60000)%250==0:
                diagnostic,predicted,targets,errors=diagnostics(actor,features,labels,span,data,strata,pairs,step)
                diagnostic['elapsed_seconds']=time.perf_counter()-started;history.append(diagnostic)
                write('full_fit_metrics.json',history)
                write('training_log.json',[dict(global_step=FIRST+i,learning_rate=lr,training_objective=value) for i,(lr,value) in enumerate(zip(rates,losses))])
                print(json.dumps(dict(global_step=step,objective=diagnostic['nine_cell_normalized_residual_objective'],
                    query250_rmse=diagnostic['actual_query250']['target_rmse_rad'],elapsed_seconds=diagnostic['elapsed_seconds'])),flush=True)
    except Exception as exc:
        torch.save(dict(actor_state=actor.state_dict(),optimizer_state=optimizer.state_dict(),rng=rng_save(),
            feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),joint_span=torch.from_numpy(span),
            completed_steps=completed,attempted_step=step,request=request),DEST/'failed_progress.pt')
        write('fit_failure.json',dict(completed_steps=completed,attempted_step=step,exception_type=type(exc).__name__,
            message=str(exc),failed_progress_sha256=sha(DEST/'failed_progress.pt'),ordinary_final_exported=False,physical_evaluations=0))
        raise
    assert completed==LAST and len(losses)==len(rates)==5000 and len(history)==21
    assert same_tree(rng_save(),saved['rng'])
    assert all(int(state['step'])==65000 for state in optimizer.state.values())
    actor.eval()
    checkpoint=dict(kind=KIND,actor_state=actor.state_dict(),optimizer_state=optimizer.state_dict(),rng=rng_save(),
        feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),joint_span=torch.from_numpy(span),
        completed_steps=65000,additional_updates=5000,request=request)
    torch.save(checkpoint,DEST/'student_head.pt')
    np.savez_compressed(DEST/'teacher_fit.npz',predicted_delta=predicted,predicted_applied_target=targets,
        expert_target=data['expert_target'],applied_error=errors,feature_mean=mean,feature_std=std,
        dataset=datasets,control=data['control'],source_frame=data['source_frame'])
    np.savez_compressed(DEST/'training_arrays.npz',global_step=np.arange(FIRST,LAST+1),
        training_objective=np.asarray(losses),learning_rate=np.asarray(rates))
    final_objective=diagnostic['nine_cell_normalized_residual_objective']
    improved=math.isfinite(final_objective) and final_objective<before['nine_cell_normalized_residual_objective']
    write('optimization_completed.json',dict(global_step=65000,additional_updates=5000,
        checkpoint_sha256=sha(DEST/'student_head.pt'),fit_arrays_sha256=sha(DEST/'teacher_fit.npz'),
        training_arrays_sha256=sha(DEST/'training_arrays.npz'),full_objective_improved=improved,
        export_validation_pending=True,physical_evaluations=0))
    export(actor,mean,std,span,DEST/'student_head.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(DEST/'student_head.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    exported=session.run(None,{'features':X})[0];maximum=float(np.max(np.abs(exported-predicted)))
    parity=bool(np.isfinite(exported).all() and maximum<1e-5)
    result=dict(kind=KIND,experiment=request['experiment'],steps=65000,additional_updates=5000,
        stop_reason='fixed5000 updates completed; ordinaryfinal65000',network_training_complete=True,
        initial_nine_cell_objective=before['nine_cell_normalized_residual_objective'],final_nine_cell_objective=final_objective,
        full_objective_improved=improved,ONNX_vs_Torch_max_delta_rad=maximum,export_parity_passed=parity,
        rollout_numerical_prerequisites_pass=improved and parity,final_fit_diagnostic=diagnostic,
        checkpoints={p.name:sha(p) for p in (DEST/'student_head.onnx',DEST/'student_head.pt')},
        restoration_parity_sha256=sha(DEST/'restoration60000_parity.json'),elapsed_seconds=time.perf_counter()-started,
        closed_loop_assessment_pending=True,physical_evaluations=0,expert_queries=0,hardware_authorized=False)
    write('report.json',result)
    assert improved and parity,('ordinary final numerical gate failed',improved,maximum)
    print(json.dumps(dict(training_complete=True,ordinary_final_global_step=65000,objective=final_objective,
        onnx_max_difference=maximum,head_sha256=sha(DEST/'student_head.onnx'))),flush=True)

if __name__=='__main__':main()
