"""Independent saved evidence audit of the proposed70000-to75000 continuation.

Requires the actual final training-receipt hash. No head evaluation, trainer
helper import, optimizer step, ORT session, BFM inference or physics is performed.
All physical labels must already have their independent completed-data review.
"""
import argparse
import json
import math
import re
import sys
import traceback
from pathlib import Path
import numpy as np
import torch
import onnx
from onnx import numpy_helper
from audit_physical_fit_math import (Proof,sha,read,archive,local,budgets,phase_cells,
    reconstruct_physical,nominal_metrics,velocity_metrics,physical_metrics,logged_cell_reductions,compose_three)

def has_hash(value,digest):
    if isinstance(value,dict):return any(has_hash(v,digest) for v in value.values())
    if isinstance(value,list):return any(has_hash(v,digest) for v in value)
    return value==digest

def main(args,p):
    p.check('required receipt sha256',re.fullmatch('[0-9a-f]{64}',args.training_receipt_sha256) is not None)
    p.check('Windows audit runtime',sys.platform=='win32' and sys.version_info[:3]==(3,10,11))
    p.check('pinned numerical packages',np.__version__=='1.23.5' and torch.__version__=='2.10.0+cpu' and onnx.__version__=='1.22.0')
    torch.set_num_threads(1)
    global_rng=torch.get_rng_state().clone()
    experiment=args.experiment.resolve();fit=experiment/'fit'
    prior=experiment.parent/'velocity_chord_student_v1';generation=prior/'generation'
    pins={}
    def pin(path,expected=None):
        path=local(path).resolve();digest=sha(path)
        if expected is not None:p.check('pin '+str(path),digest==expected)
        if str(path) in pins:p.check('repeated pin unchanged '+str(path),pins[str(path)]==digest)
        pins[str(path)]=digest
        return path
    receipt_path=pin(experiment/'training_frozen_inputs.json',args.training_receipt_sha256)
    receipt=read(receipt_path)
    source_dir=local(receipt['source_directory'])
    for name,digest in receipt['source_sha256'].items():pin(source_dir/name,digest)
    for name,digest in receipt['input_sha256'].items():pin(name,digest)
    for path in (Path(__file__),Path(__file__).with_name('audit_physical_fit_math.py')):pin(path)
    request_path=pin(experiment/'training_request.json',receipt['training_request_sha256'])
    training=read(request_path)
    p.tree(training['input_sha256'],receipt['input_sha256'],'training input map')
    p.check('selected fixed objective',training['kind']=='one_physical_response_continuation_from70000')
    p.tree(training['coefficients'],dict(nominal=1.,velocity=1.,physical=1.),'unit coefficients')
    p.check('fixed update range',training['additional_updates']==5000 and training['ordinary_final_step']==75000)
    clear_path=pin(experiment/'training_clearance.json');clear=read(clear_path)
    for key in ('approved','model_fitting_authorized'):p.check('clearance '+key,clear[key] is True)
    p.check('clearance count',clear['additional_updates']==5000 and clear['ordinary_final_step']==75000)
    p.check('clearance receipt',clear['frozen_receipt_sha256']==sha(receipt_path))
    p.check('clearance training request',clear['training_request_sha256']==sha(request_path))
    p.check('required review roles',set(training['reviews'])=={'prior_fit','prior_export','branch_data','source'})
    for role,spec in training['reviews'].items():
        rpath=pin(spec['path'],spec['sha256']);review=read(rpath)
        p.check(role+' review passed',review[spec['pass_field']] is True)
        p.check(role+' review has subjects',bool(spec['subjects']))
        for name,digest in spec['subjects'].items():
            pin(name,digest);p.check(role+' direct subject binding '+name,has_hash(review,digest))
    p.check('source review role bound',clear['source_review_sha256']==training['reviews']['source']['sha256'])
    physical_paths=training['physical_paths']
    manifest_path=pin(physical_paths['manifest']);manifest=read(manifest_path)
    producer_path=pin(physical_paths['report']);producer=read(producer_path)
    collection_request=pin(physical_paths['collection_request'])
    p.check('branch data reviewer directly binds manifest',
        training['reviews']['branch_data']['subjects'][str(physical_paths['manifest'])]==sha(manifest_path))
    p.check('producer complete',producer['completed'] is True and producer['passed'] is True and manifest['complete'] is True)
    p.check('producer all requested3054',producer['rows']==producer['nominal_verified']==producer['policy_branches']==manifest['rows']==3054)
    p.check('producer manifest binding',producer['manifest_sha256']==sha(manifest_path))
    p.check('producer request binding',producer['request_sha256']==manifest['request_sha256']==sha(collection_request))
    p.check('producer actual251 witness',producer['query250_actual251_calibration_passed'] is True)
    p.check('producer source rehash',producer['all_frozen_inputs_unchanged'] is True)
    arrays={}
    for name,spec in manifest['arrays'].items():
        path=(manifest_path.parent/spec['path']).resolve()
        p.check('array contained '+name,path.parent==manifest_path.parent.resolve())
        pin(path,spec['sha256'])
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        p.check('manifest array schema '+name,list(a.shape)==spec['shape'] and str(a.dtype)==spec['dtype'] and len(a)==3054)
        arrays[name]=a
    names=('student_head.pt','student_head.onnx','training_arrays.npz','sampled_center_rows.npy','sampled_axes.npy',
        'initial_predictions.npz','final_predictions.npz','initial_physical_predictions.npz','final_physical_predictions.npz',
        'initial_chord_metrics.json','final_chord_metrics.json','initial_nominal_metrics.json','final_nominal_metrics.json',
        'initial_physical_metrics.json','final_physical_metrics.json','physical_row_selection.npz','physical_coverage.json',
        'report.json','request.json','restoration70000_parity.json','restored70000.onnx','optimization_completed.json',
        'attempt_status.json','training_progress.npy','training_cell_losses.npy',
        'initial_sensitivity.json','final_sensitivity.json','initial_jacobians.npz','final_jacobians.npz')
    for name in names:pin(fit/name)
    for name in ('student_head.pt','student_head.onnx','report.json','final_predictions.npz'):pin(prior/'fit'/name)
    for name in ('centers.npz','base_target.npy','teacher_target.npy','report.json'):pin(generation/name)
    pair_path=pin(experiment.parent/'bfm_entry250_labels_v1/compatibility/fixed_pairs.json')
    request=dict(kind='independent_saved_physical_fit_evidence',source_sha256=sha(__file__),
        helper_sha256=sha(Path(__file__).with_name('audit_physical_fit_math.py')),input_sha256=pins.copy(),
        required_training_receipt_sha256=args.training_receipt_sha256,
        private_generator_calls=45000,private_generator_row_axis_pairs=2880000,
        model_evaluations=0,ORT_calls=0,optimizer_updates=0,physics_steps=0,
        runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,onnx=onnx.__version__,threads=1))
    (args.output/'request.json').write_text(json.dumps(request,indent=2)+'\n',encoding='utf-8')
    audit_request_sha=sha(args.output/'request.json')
    p.stage='physical identities and saved supervision'
    centers=archive(generation/'centers.npz')
    p.exact(centers['dataset'],np.repeat(np.arange(3,dtype=np.int64),1019),'3057 dataset order')
    p.exact(centers['control'],np.tile(np.arange(250,1269,dtype=np.int64),3),'3057 controls')
    p.exact(centers['source_frame'],centers['control']+11,'3057 source frames')
    physical=reconstruct_physical(arrays,centers,p)
    V=len(physical['rows']);expected=budgets(V)
    p.check('nonempty actual qualified labels',0<V<=3054)
    p.check('valid counts bound',training['valid_rows']==producer['labels_valid']==manifest['labels_valid']==V)
    p.check('strict failure counts retained',producer['strict_failed']==manifest['strict_failed']==3054-V)
    p.tree(training['budgets'],expected,'independent budgets')
    calls={name:dict(attempted=n,returned=n) for name,n in dict(backward=V,actor=V,head=3054+V).items()}
    p.tree(producer['graph_calls'],calls,'producer graph accounting')
    p.tree(manifest['graph_calls'],calls,'manifest graph accounting')
    total_steps=30540+int(np.sum(arrays['policy_returned_steps']))
    p.check('cumulative native step accounting',producer['native_attempted']==producer['native_returned']==total_steps<=61080)
    if 'prefix_native_steps' in producer:
        p.check('continued prefix explicit',producer['prefix_nominal_rows']==2969 and producer['prefix_native_steps']==29690)
        p.check('continued current steps',producer['current_attempt_native_steps']==total_steps-29690)
    selected=archive(fit/'physical_row_selection.npz')
    for key,value in dict(requested_valid_mask=physical['valid_mask'],selected_rows=physical['rows'],successor_center=physical['successor']).items():
        p.exact(selected[key],value,'fitted row map '+key)
    p.tree(read(fit/'physical_coverage.json'),physical['coverage'],'independent coverage')
    p.stage='checkpoint and export identity'
    report=read(fit/'report.json');fit_request=read(fit/'request.json');status=read(fit/'attempt_status.json')
    for key in ('completed','optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed'):
        p.check('completed fit '+key,report[key] is True)
    p.check('ordinary75000',report['ordinary_final_step']==75000 and report['additional_updates']==5000)
    p.check('report head identity',report['checkpoint_sha256']==sha(fit/'student_head.pt') and report['onnx_sha256']==sha(fit/'student_head.onnx'))
    p.check('fit request range',(fit_request['first_step'],fit_request['last_step'],fit_request['updates'])==(70001,75000,5000))
    p.check('fit request receipt',fit_request['frozen_receipt_sha256']==sha(receipt_path))
    p.check('fit request clearance',fit_request['clearance_sha256']==sha(clear_path))
    p.check('fit request dynamic inputs',fit_request['training_request_sha256']==sha(request_path))
    p.tree(fit_request['input_request'],training,'saved full input request')
    p.tree(fit_request['budgets'],expected,'fit request budgets')
    p.tree(fit_request['physical_coverage'],physical['coverage'],'fit request coverage')
    p.check('fit physical requested denominators',fit_request['physical_requested_denominators']==[99,819,100]*3)
    p.check('fit unit added coefficients',fit_request['lambda_chord']==fit_request['lambda_physical']==1.)
    p.check('prior checkpoint bound',fit_request['prior_sha256']==sha(prior/'fit/student_head.pt'))
    before=torch.load(prior/'fit/student_head.pt',map_location='cpu',weights_only=True)
    after=torch.load(fit/'student_head.pt',map_location='cpu',weights_only=True)
    p.check('saved checkpoint stages',before['completed_steps']==70000 and after['completed_steps']==75000)
    p.tree(after['request'],fit_request,'checkpoint request')
    for name in ('feature_mean','feature_std','joint_span'):p.exact(after[name].numpy(),before[name].numpy(),'unchanged '+name)
    p.check('six optimizer states',len(before['optimizer_state']['state'])==len(after['optimizer_state']['state'])==6)
    p.tree(after['optimizer_state']['param_groups'],before['optimizer_state']['param_groups'],'restored/final optimizer hyperparameters')
    for state in before['optimizer_state']['state'].values():p.check('prior optimizer step70000',int(state['step'])==70000)
    for key,state in after['optimizer_state']['state'].items():
        p.check('final optimizer step75000 '+str(key),int(state['step'])==75000)
        p.check('finite optimizer '+str(key),all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v)))
    for key,a in after['actor_state'].items():p.check('finite parameter '+key,bool(torch.isfinite(a).all()))
    p.check('ordinary head updated',any(not torch.equal(a,before['actor_state'][k]) for k,a in after['actor_state'].items()))
    exported=onnx.load(fit/'student_head.onnx');prior_onnx=onnx.load(prior/'fit/student_head.onnx')
    tensors={a.name:numpy_helper.to_array(a) for a in exported.graph.initializer}
    p.check('expected ONNX initializer names',set(tensors)=={'mean','std','span','w0','w1','w2','b0','b1','b2'})
    for name,key in (('mean','feature_mean'),('std','feature_std'),('span','joint_span')):p.exact(tensors[name],after[key].numpy(),'export '+name)
    for n in range(3):
        p.exact(tensors['w'+str(n)],after['actor_state'][str(2*n)+'.weight'].numpy().T,'export weight'+str(n))
        p.exact(tensors['b'+str(n)],after['actor_state'][str(2*n)+'.bias'].numpy(),'export bias'+str(n))
    del exported.graph.initializer[:];del prior_onnx.graph.initializer[:]
    p.check('unchanged ONNX graph',exported.SerializeToString(deterministic=True)==prior_onnx.SerializeToString(deterministic=True))
    p.check('restored ONNX byte exact',sha(fit/'restored70000.onnx')==sha(prior/'fit/student_head.onnx'))
    p.stage='all sampling and logged updates'
    p.tree(before['rng']['numpy'],after['rng']['numpy'],'NumPy RNG unchanged')
    cells=phase_cells(centers['dataset'],centers['control'])
    rows=np.load(fit/'sampled_center_rows.npy',allow_pickle=False);axes=np.load(fit/'sampled_axes.npy',allow_pickle=False)
    p.check('sample schema',rows.shape==axes.shape==(5000,576) and rows.dtype==np.int32 and axes.dtype==np.int8)
    generator=torch.Generator(device='cpu');generator.set_state(before['rng']['torch'])
    for update in range(5000):
        p.context=dict(update=np.int64(update))
        for cell,indices in enumerate(cells):
            draws=torch.randint(0,len(indices)*23,(64,),generator=generator).numpy()
            part=slice(cell*64,(cell+1)*64)
            p.exact(rows[update,part],indices[draws//23].astype(np.int32),'sample rows '+str((update,cell)))
            p.exact(axes[update,part],(draws%23).astype(np.int8),'sample axes '+str((update,cell)))
    p.exact(generator.get_state().numpy(),after['rng']['torch'].numpy(),'private final Torch RNG')
    arrays_fit=archive(fit/'training_arrays.npz')
    p.exact(arrays_fit['global_step'],np.arange(70001,75001),'all5000 global steps')
    rates=np.asarray([3e-7+.5*(3e-6-3e-7)*(1+math.cos(math.pi*k/4999)) for k in range(5000)])
    p.exact(arrays_fit['learning_rate'],rates,'all5000 cosine rates')
    components=('nominal_objective','sampled_chord_objective','physical_objective','training_objective')
    for key in components:
        a=arrays_fit[key];p.check('finite nonnegative5000 '+key,a.shape==(5000,) and a.dtype==np.float64 and np.isfinite(a).all() and (a>=0).all())
    p.exact(compose_three(arrays_fit['nominal_objective'],arrays_fit['sampled_chord_objective'],arrays_fit['physical_objective']),
        arrays_fit['training_objective'],'three-term loss composition')
    logged_cells=arrays_fit['cell_losses']
    p.check('logged cells schema',logged_cells.shape==(5000,3,9) and logged_cells.dtype==np.float64 and np.isfinite(logged_cells).all())
    p.exact(np.load(fit/'training_cell_losses.npy',allow_pickle=False),logged_cells,'periodic/final cell arrays')
    p.exact(np.load(fit/'training_progress.npy',allow_pickle=False),np.stack([arrays_fit[k] for k in (*components,'learning_rate')],axis=1),'final progress columns')
    for k in range(5000):
        reductions=logged_cell_reductions(logged_cells[k])
        for term,value in enumerate(reductions):
            p.check('cell reduction '+str((k,term)),value==float(arrays_fit[components[term]][k]))
    p.context={}
    optimizer_report=read(fit/'optimization_completed.json')
    p.check('optimization final ordinary',optimizer_report['ordinary_final_step']==75000)
    p.check('optimization checkpoint hash',optimizer_report['checkpoint_sha256']==sha(fit/'student_head.pt'))
    p.check('optimization arrays hash',optimizer_report['training_arrays_sha256']==sha(fit/'training_arrays.npz'))
    p.stage='initial final all saved predictions'
    span=before['joint_span'].numpy();p.exact(span,centers['joint_span'],'original spans')
    probe_base=np.load(generation/'base_target.npy',mmap_mode='r',allow_pickle=False)
    probe_target=np.load(generation/'teacher_target.npy',mmap_mode='r',allow_pickle=False)
    prior_predictions=archive(prior/'fit/final_predictions.npz');pairs=read(pair_path)
    summaries={}
    for stage,step in (('initial',70000),('final',75000)):
        prediction=archive(fit/(stage+'_predictions.npz'))
        physical_prediction=archive(fit/(stage+'_physical_predictions.npz'))
        for key in ('predicted_delta','onnx_predicted_delta'):
            a=prediction[key];p.check(stage+' legacy '+key,a.shape==(143679,23) and a.dtype==np.float32 and np.isfinite(a).all())
            a=physical_prediction[key];p.check(stage+' physical '+key,a.shape==(V,23) and a.dtype==np.float32 and np.isfinite(a).all())
        for key in ('actual_seven_delta','actual_seven_onnx_delta'):
            a=prediction[key];p.check(stage+' '+key,a.shape==(7,23) and a.dtype==np.float32 and np.isfinite(a).all())
        normalized=prediction['normalized_centers']
        p.check(stage+' normalized centers',normalized.shape==(3057,23) and normalized.dtype==np.float32 and np.isfinite(normalized).all())
        p.exact((torch.from_numpy(normalized)*torch.from_numpy(span)).numpy(),prediction['predicted_delta'][:3057],stage+' normalized/physical residual')
        legacy_parity=float(max(np.max(np.abs(prediction['predicted_delta']-prediction['onnx_predicted_delta'])),
            np.max(np.abs(prediction['actual_seven_delta']-prediction['actual_seven_onnx_delta']))))
        physical_parity=float(np.max(np.abs(physical_prediction['predicted_delta']-physical_prediction['onnx_predicted_delta'])))
        parity=max(legacy_parity,physical_parity);p.check(stage+' export numerical parity',parity<1e-5)
        nominal=nominal_metrics(prediction,centers,cells,span,pairs,step)
        velocity=velocity_metrics(prediction['predicted_delta'],centers,probe_base,probe_target,cells,span)
        physical_result=physical_metrics(prediction['predicted_delta'][:3057],physical_prediction['predicted_delta'],centers,physical,span)
        if stage=='initial':
            physical_result['restored_Windows_ORT_vs_collector_WSL_batch1_max_delta']=float(np.max(np.abs(
                physical_prediction['onnx_predicted_delta']-physical['collector_head_delta'])))
            physical_result['cross_platform_comparison_is_not_byte_parity']=True
            p.check('initial all legacy keys',set(prediction)==set(prior_predictions))
            for key in prediction:p.exact(prediction[key],prior_predictions[key],'all restored70000 legacy '+key)
            prior_report=read(prior/'fit/report.json')
            p.check('initial nominal matches70000',nominal['nine_cell_nominal_objective']==prior_report['final_nominal_objective'])
            p.check('initial velocity matches70000',velocity['full_nine_cell_chord_objective']==prior_report['final_chord_objective'])
        else:p.check('reported final full parity',parity==report['final_head_parity'])
        p.tree(nominal,read(fit/(stage+'_nominal_metrics.json')),stage+' all nominal metrics')
        p.tree(velocity,read(fit/(stage+'_chord_metrics.json')),stage+' all velocity metrics')
        p.tree(physical_result,read(fit/(stage+'_physical_metrics.json')),stage+' all physical metrics')
        p.check(stage+' nominal report',nominal['nine_cell_nominal_objective']==report[stage+'_nominal_objective'])
        p.check(stage+' velocity report',velocity['full_nine_cell_chord_objective']==report[stage+'_chord_objective'])
        p.check(stage+' physical report',physical_result['nine_cell_response_objective']==report[stage+'_physical_objective'])
        summaries[stage]=dict(nominal=nominal['nine_cell_nominal_objective'],velocity=velocity['full_nine_cell_chord_objective'],
            physical_response=physical_result['nine_cell_response_objective'],physical_absolute=physical_result['nine_cell_absolute_branch_diagnostic'],
            export_max_delta_rad=parity)
    p.stage='all budgets and final hashes'
    p.tree(report['physical_coverage'],physical['coverage'],'final report coverage')
    p.check('report branch manifest',report['physical_data_manifest_sha256']==sha(manifest_path))
    counts=report['attempt_counters']
    for record_name,record in (('report',counts),('status',status)):
        p.check(record_name+' completed75000',record['completed_steps']==75000 and record['attempted_step']==75000)
        p.check(record_name+' all optimizer steps',record['optimizer_step_counters']==[75000]*6)
        p.check(record_name+' all committed updates',record['committed_sample_rows']==record['committed_loss_rows']==5000)
        for name,total in (('head_onnx_calls',expected['head_onnx_calls']),('diagnostic_torch_rows',expected['diagnostic_torch_rows']),
            ('training_head_rows',expected['training_head_rows']),('analytical_head_evaluations',14)):
            p.check(record_name+' '+name,record[name+'_attempted']==record[name+'_returned']==total)
        for suffix in ('attempted','returned'):
            p.check(record_name+' old1126 '+suffix,record['legacy_diagnostics']['head_onnx_calls_'+suffix]==1126)
            p.check(record_name+' physical calls '+suffix,record['physical_diagnostics']['head_onnx_calls_'+suffix]==expected['physical_head_onnx_calls'])
            p.check(record_name+' old Torch rows '+suffix,record['legacy_diagnostics']['diagnostic_torch_rows_'+suffix]==287372)
            p.check(record_name+' physical Torch rows '+suffix,record['physical_diagnostics']['diagnostic_torch_rows_'+suffix]==2*V)
    p.check('status complete',status['stage']=='COMPLETE')
    p.check('reported total calls',report['head_ONNX_calls']==expected['head_onnx_calls'])
    restoration=read(fit/'restoration70000_parity.json')
    for key in ('full_model_AdamW_RNG_exact','all3057_nominal_predictions_and_loss_exact','old_normalization_span_exact','restored_ONNX_bytes_exact'):
        p.check('restoration report '+key,restoration[key] is True)
    p.check('initial counted calls',restoration['head_ONNX_calls']==expected['head_onnx_calls']//2)
    p.check('audit global RNG unchanged',torch.equal(global_rng,torch.get_rng_state()))
    for path,digest in pins.items():p.check('final unchanged '+path,sha(path)==digest)
    p.check('audit request unchanged',sha(args.output/'request.json')==audit_request_sha)
    result=dict(kind='independent_saved_physical_fit_evidence',passed=True,comparisons=p.count,
        all2880000_private_sampler_pairs_and_final_Torch_RNG_exact=True,NumPy_RNG_unchanged=True,
        all5000_learning_rates_three_loss_compositions_and_cell_reductions_exact=True,
        ordinary_final_optimizer_steps=75000,original_norm_span_exact=True,ONNX_weights_and_graph_exact=True,
        all3054_physical_identities_validity_and_requested_cell_weights_exact=True,
        initial_all_legacy_predictions_byteexact_to70000=True,initial_final_all_prediction_metrics_exact=True,
        fixed_first24_requested_windows_and_clip_replan_zero_gain_groups_exact=True,
        requested_physical_rows=3054,valid_physical_rows=V,strict_failed_rows=3054-V,coverage=physical['coverage'],
        budgets=expected,summaries=summaries,all_inputs_and_sources_unchanged=True,
        model_evaluations=0,ORT_calls=0,optimizer_updates=0,physics_steps=0,
        limitation='Saved-evidence audit only. Optimizer updates, graph outputs and native dynamics are not reexecuted. Analytical sensitivity values are pinned/counted, not recomputed. This is not behavioral qualification.',
        request_sha256=audit_request_sha,source_sha256=sha(__file__),helper_sha256=request['helper_sha256'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--training-receipt-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ('audit_physical_fit_evidence.py','audit_physical_fit_math.py'):
        (args.output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    proof=Proof(args.output)
    try:main(args,proof)
    except BaseException as error:
        if proof.context:np.savez_compressed(args.output/'failed_context.npz',**proof.context)
        failure=dict(passed=False,exception=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            stage=proof.stage,comparisons_completed=proof.count,required_training_receipt_sha256=args.training_receipt_sha256,
            source_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('audit_physical_fit_math.py')),
            model_evaluations=0,ORT_calls=0,optimizer_updates=0,physics_steps=0)
        (args.output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
        raise
    finally:proof.finish()
