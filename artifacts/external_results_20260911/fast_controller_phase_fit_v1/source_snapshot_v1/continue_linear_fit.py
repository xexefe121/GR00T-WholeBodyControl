"""One deterministic continuation of the existing nominal supervised fit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import torch

from fit_linear_head import CONFIG, make_actor, export
from student_linear_runtime import BASE,KIND,FROZEN_RECEIPT,assert_frozen,archive,sha

ORIGINAL=BASE.parent/'fast_controller_nominal_pilot_v1'
DEST=BASE/'fit'
MAX_STEPS=20000
CHECK_EVERY=1000
FIT_STOP=dict(full1269_applied_target_RMSE_at_most_rad=.01,
    maximum_first24_perjoint_absolute_error_p95_at_most_rad=.03,
    logic='both at the same fixed1000-step diagnostic; no physical acceptance implied')


def write(name,value):
    (DEST/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def setup():
    assert_frozen();torch.set_num_threads(1)
    torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed'])
    d=archive(BASE/'labels/labels.npz')
    assert sha(BASE/'labels/labels.npz')==sha(ORIGINAL/'labels/labels.npz')
    X=d['features'].astype(np.float32);span=d['joint_span'].astype(np.float32)
    Y=(d['residual_rad']/span).astype(np.float32)
    mean=X.mean(0).astype(np.float32)
    std=np.maximum(X.std(0),CONFIG['feature_std_floor']).astype(np.float32)
    features=torch.from_numpy((X-mean)/std);labels=torch.from_numpy(Y)
    actor=make_actor()
    optimizer=torch.optim.AdamW(actor.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    strata=[np.arange(0,250),np.arange(250,350),np.arange(350,1169),np.arange(1169,1269)]
    return d,X,span,mean,std,features,labels,actor,optimizer,strata


def one_step(actor,optimizer,features,labels,strata):
    picked=np.concatenate([np.random.choice(ids,64,replace=True) for ids in strata])
    prediction=actor(features[picked]);loss=torch.mean((prediction-labels[picked])**2)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(),CONFIG['gradient_clip_norm']);optimizer.step()
    return float(loss.detach())


def assess_fit(actor,features,span,d,strata,step):
    # ELU/linear layers have no train/eval-dependent state, dropout or sampling.
    with torch.inference_mode():predicted=(actor(features)*torch.from_numpy(span)).numpy()
    target=np.clip(d['base_target']+predicted,d['joint_limits'][:,0],d['joint_limits'][:,1])
    error=target-d['expert_target'];residual_error=predicted-d['residual_rad']
    phases={}
    for name,indices in zip(('initial_entry','acquisition','source','return'),strata):
        e=error[indices]
        phases[name]=dict(controls=len(indices),applied_target_rmse_rad=float(np.sqrt(np.mean(e**2))),
            applied_target_rmse_by_joint_rad=np.sqrt(np.mean(e**2,axis=0)).tolist(),
            applied_target_abs_error_p95_by_joint_rad=np.percentile(np.abs(e),95,axis=0).tolist(),
            preclip_residual_rmse_rad=float(np.sqrt(np.mean(residual_error[indices]**2))),
            applied_target_max_error_rad=float(np.max(np.abs(e))))
    rmse=float(np.sqrt(np.mean(error**2)))
    p95=np.percentile(np.abs(error[:24]),95,axis=0)
    diagnostic=dict(step=step,full1269_applied_target_RMSE_rad=rmse,
        full1269_perjoint_applied_target_RMSE_rad=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        full1269_perjoint_absolute_error_p95_rad=np.percentile(np.abs(error),95,axis=0).tolist(),
        first24_perjoint_absolute_error_p95_rad=p95.tolist(),
        maximum_first24_perjoint_absolute_error_p95_rad=float(p95.max()),
        initial_perjoint_signed_target_error_rad=error[0].tolist(),
        teacher_state_training_metrics=phases,
        fit_stop_criteria_met=bool(rmse<=.01 and p95.max()<=.03),
        fit_metrics_not_physical_acceptance=True)
    return diagnostic,predicted,target,error


def rng_save():
    name,state,pos,has_gauss,cached=np.random.get_state()
    return dict(torch=torch.get_rng_state(),numpy=dict(name=name,state=state.tolist(),pos=pos,has_gauss=has_gauss,cached=cached))


def rng_restore(rng):
    torch.set_rng_state(rng['torch']);n=rng['numpy']
    np.random.set_state((n['name'],np.asarray(n['state'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached']))


def reconstruction():
    assert not (DEST/'reconstruction_1000.pt').exists()
    d,X,span,mean,std,features,labels,actor,optimizer,strata=setup()
    parity=json.loads((ORIGINAL/'zero_parity/report.json').read_text())
    assert parity['all_control_state_history_rawaction_target_bitexact'] and parity['native2msstrict_pass']
    request=dict(kind=KIND,experiment='one_continued_nominal_supervised_fit_v1',
        original_configuration=CONFIG,total_step_limit=MAX_STEPS,diagnostic_interval=CHECK_EVERY,fit_stop=FIT_STOP,
        seed=773,threads=1,configuration_changes=['maximum total steps and declared fit-only early stop'],
        unchanged='labels/features/normalization/architecture/span/outputbounds/phasebatches/AdamW/seed/runtime/physics',
        previous_fit_checkpoint_sha256=sha(ORIGINAL/'fit/student_head.pt'),
        original_frozen_sources_sha256=sha(ORIGINAL/'frozen_inputs_v2.json'),
        frozen_sources_sha256=sha(FROZEN_RECEIPT),labels_sha256=sha(BASE/'labels/labels.npz'),
        prior_zero_preflight_report_sha256=sha(ORIGINAL/'zero_parity/report.json'),
        no_dagger_labels=True,no_hardware=True,training_metric_is_not_validation=True)
    write('request.json',request)
    export(actor,mean,std,span,DEST/'reconstructed_zero_initialization.onnx')
    assert sha(DEST/'reconstructed_zero_initialization.onnx')==sha(DEST/'zero_head.onnx')
    started=time.perf_counter();records=[]
    for step in range(1,1001):
        loss=one_step(actor,optimizer,features,labels,strata)
        if step%100==0:
            records.append(dict(step=step,normalized_residual_training_loss=loss,seconds=time.perf_counter()-started))
    prior=torch.load(ORIGINAL/'fit/student_head.pt',map_location='cpu',weights_only=True)
    tensor_parity={k:bool(torch.equal(v,prior['actor_state'][k])) for k,v in actor.state_dict().items()}
    assert all(tensor_parity.values()),tensor_parity
    for name,array in [('feature_mean',mean),('feature_std',std),('joint_span',span)]:
        assert torch.equal(torch.from_numpy(array),prior[name]),name
    oldrecords=json.loads((ORIGINAL/'fit/metrics.json').read_text())
    assert [r['normalized_residual_training_loss'] for r in records]==[r['normalized_residual_training_loss'] for r in oldrecords]
    diagnostic,predicted,target,error=assess_fit(actor,features,span,d,strata,1000)
    oldfit=archive(ORIGINAL/'fit/teacher_fit.npz')
    np.testing.assert_array_equal(predicted,oldfit['predicted_delta'])
    np.testing.assert_array_equal(target,oldfit['predicted_applied_target'])
    state=dict(completed_steps=1000,actor_state=actor.state_dict(),optimizer_state=optimizer.state_dict(),
        rng=rng_save(),feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),
        joint_span=torch.from_numpy(span),request=json.loads(json.dumps(request)))
    torch.save(state,DEST/'reconstruction_1000.pt')
    receipt=dict(completed_steps=1000,all_model_tensors_bit_exact=True,per_tensor_exact=tensor_parity,
        normalization_and_span_bit_exact=True,all_ten_original_logged_losses_bit_exact=True,
        full_teacher_prediction_arrays_bit_exact=True,optimizer_and_rng_reconstructed=True,
        continuation_steps_launched=0,elapsed_seconds=time.perf_counter()-started,
        checkpoint_sha256=sha(DEST/'reconstruction_1000.pt'),original_checkpoint_sha256=sha(ORIGINAL/'fit/student_head.pt'),
        request_sha256=sha(DEST/'request.json'),frozen_sources_sha256=sha(FROZEN_RECEIPT))
    write('reconstruction_metrics.json',records);write('full_fit_metrics.json',[diagnostic]);write('reconstruction_parity.json',receipt)
    print(json.dumps(receipt),flush=True)


def continuation():
    assert not (DEST/'student_head.pt').exists() and not (DEST/'report.json').exists()
    d,X,span,mean,std,features,labels,actor,optimizer,strata=setup()
    parity=json.loads((DEST/'reconstruction_parity.json').read_text())
    assert parity['all_model_tensors_bit_exact'] and parity['continuation_steps_launched']==0
    assert sha(DEST/'reconstruction_1000.pt')==parity['checkpoint_sha256']
    saved=torch.load(DEST/'reconstruction_1000.pt',map_location='cpu',weights_only=True)
    actor.load_state_dict(saved['actor_state']);optimizer.load_state_dict(saved['optimizer_state']);rng_restore(saved['rng'])
    assert all(torch.equal(v,saved['actor_state'][k]) for k,v in actor.state_dict().items())
    # Loading and copying diagnostics consumes no RNG. Capture the resumed state
    # to document that both generators resume where exact reconstruction ended.
    now=rng_save();assert torch.equal(now['torch'],saved['rng']['torch']) and now['numpy']==saved['rng']['numpy']
    records=json.loads((DEST/'reconstruction_metrics.json').read_text())
    diagnostics=json.loads((DEST/'full_fit_metrics.json').read_text())
    started=time.perf_counter()
    for step in range(1001,MAX_STEPS+1):
        loss=one_step(actor,optimizer,features,labels,strata)
        if step%100==0:
            records.append(dict(step=step,normalized_residual_training_loss=loss,continuation_seconds=time.perf_counter()-started))
        if step%CHECK_EVERY==0:
            diagnostic,predicted,target,error=assess_fit(actor,features,span,d,strata,step)
            diagnostic['continuation_seconds']=time.perf_counter()-started;diagnostics.append(diagnostic)
            write('metrics.json',records);write('full_fit_metrics.json',diagnostics)
            print(json.dumps({k:diagnostic[k] for k in ('step','full1269_applied_target_RMSE_rad','maximum_first24_perjoint_absolute_error_p95_rad','fit_stop_criteria_met','continuation_seconds')}),flush=True)
            if diagnostic['fit_stop_criteria_met']:break
    actor.eval()
    export(actor,mean,std,span,DEST/'student_head.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(DEST/'student_head.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    exported=session.run(None,{'features':X})[0]
    delta=float(np.max(np.abs(exported-predicted)))
    assert delta<1e-5,(delta,'ONNX export mismatch')
    zero=ort.InferenceSession(str(DEST/'zero_head.onnx'),sess_options=options,providers=['CPUExecutionProvider']).run(None,{'features':X})[0]
    np.testing.assert_array_equal(zero,np.zeros_like(zero))
    request=json.loads((DEST/'request.json').read_text())
    checkpoint=dict(kind=KIND,actor_state=actor.state_dict(),feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),
        joint_span=torch.from_numpy(span),completed_steps=step,request=json.loads(json.dumps(request)),
        optimizer_state=optimizer.state_dict(),rng=rng_save())
    torch.save(checkpoint,DEST/'student_head.pt')
    np.savez_compressed(DEST/'teacher_fit.npz',predicted_delta=predicted,predicted_applied_target=target,
        expert_target=d['expert_target'],applied_error=error,feature_mean=mean,feature_std=std,source_frame=d['source_frame'])
    result=dict(kind=KIND,experiment='one_continued_nominal_supervised_fit_v1',steps=step,maximum_total_steps=MAX_STEPS,
        stop_reason='declared fit-only thresholds met' if diagnostic['fit_stop_criteria_met'] else 'declared total20000 maximum reached',
        continuation_elapsed_seconds=time.perf_counter()-started,reconstruction_parity_sha256=sha(DEST/'reconstruction_parity.json'),
        final_fit_diagnostic=diagnostic,teacher_state_training_metrics=diagnostic['teacher_state_training_metrics'],
        ONNX_vs_Torch_max_delta_rad=delta,zero_head_all1269_teacher_states_exact=True,
        network_training_complete=True,closed_loop_assessment_pending=True,no_generalization_claim=True,
        no_intermediate_physics_evaluations=True,no_dagger_queries=True,
        checkpoints={p.name:sha(p) for p in (DEST/'zero_head.onnx',DEST/'student_head.onnx',DEST/'student_head.pt')})
    write('report.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('final_fit_diagnostic','teacher_state_training_metrics')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['reconstruct','continue'],required=True)
    stage=parser.parse_args().stage
    reconstruction() if stage=='reconstruct' else continuation()
