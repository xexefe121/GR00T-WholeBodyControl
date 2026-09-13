"""Future selected pair only; immutable data/objective, no adaptive fitting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise RuntimeError('CUBLAS configuration must precede Torch import.')
import copy
import gc
from pathlib import Path
import time
import numpy as np
import torch
from direct_contract import RETAINED,PHYSICAL_COUNTS,normalized_labels
from direct_data import read,sha
from direct_objective import nominal_loss,physical_loss
from full_state_contract import endpoint_rows,verify_schedule
from full_state_objective import full_state_loss
from context_contract import CONDITIONS,UPDATES,COEFFICIENT,BUDGETS,cosine_rate,fixed_schedule
from context_data import load_context_data,condition_data
from context_model import ContextTarget,expand_actor,assert_blinded_columns
from context_diagnostics import BACKENDS,CORPORA,run_backend,summarize,parity,save_drift,atomic
from context_promoted import from_checkpoint,export_onnx
from context_support import gate
from restoration_support import exact_saved,restore_rng
from training_support import (write,final_identity,cpu_tree,rng_save,configure_runtime,
    flatten_active,counter,counted_forward,save_loss_prefix,manifest)

BASE=Path(__file__).resolve().parent.parent

def expected_counter(calls,rows):
    return dict(calls_attempted=calls,calls_returned=calls,calls_synchronized=calls,calls_verified=calls,
        rows_attempted=rows,rows_returned=rows,rows_verified=rows)

def initial_drift(outputs,request,data):
    result={};maximum=0.
    for corpus in CORPORA:
        prior=np.load(request['restoration_predictions'][corpus],mmap_mode='r',allow_pickle=False)
        actual=np.asarray(outputs[corpus])
        if prior.shape!=actual.shape or prior.dtype!=np.float32:raise ValueError('Saved65000 prediction schema.')
        delta=(actual.astype(np.float64)-prior.astype(np.float64))*data['span'].astype(np.float64)
        value=float(np.max(np.abs(delta)))
        if not np.isfinite(value):raise ValueError('Nonfinite initial dimensional drift.')
        maximum=max(maximum,value)
        result[corpus]=dict(max_preclip_error_rad=value,RMS_preclip_error_rad=float(np.sqrt(np.mean(delta**2))),
            byte_equal=actual.tobytes()==prior.tobytes())
    return dict(passed=maximum<=1e-5,tolerance_rad=1e-5,max_preclip_error_rad=maximum,corpora=result,
        changed_MatMul_dimension=False,stored_first_layer_width=1323,first_layer_execution='split_contiguous_1000_plus_323',byte_gate_required=False)

def run_condition(condition,data,source,expanded,normalization,sc,sa,request,receipt,identity,shared):
    dest=BASE/'fit'/condition;dest.mkdir(exist_ok=False)
    counts=dict(training=counter(),diagnostics={name:counter() for name in BACKENDS},calibration_forward_calls=0,
        calibration_gradient_calls=0,BFM_calls=0,native_calls=0,manual_export_trace_calls=0)
    state=dict(condition=condition,stage='initializing',attempted_global_step=65000,completed_updates=0,
        optimizer_call_started=False,optimizer_call_returned=False,optimizer_synchronized=False)
    active={};model=None;optimizer=None;used_centers=None;used_axes=None;ledger=None
    losses=[];nc_records=[];fc_records=[];pc_records=[];outputs={name:{} for name in BACKENDS};metrics={}
    optimized=False;export_done=False;started=time.perf_counter()
    try:
        model=ContextTarget(normalization['feature_mean'],normalization['feature_std'])
        model.actor.load_state_dict(expanded,strict=True);model=model.to('cuda')
        if len(list(model.parameters()))!=6 or any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Six float32 parameter tensors required.')
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-5,foreach=False,fused=False)
        restore_rng(source['rng']);rng=rng_save()
        exact_saved(model.actor.state_dict(),expanded,'expanded initial actor');exact_saved(rng,source['rng'],'restored65000 RNG')
        if optimizer.state:raise ValueError('Fresh AdamW must be empty.')
        torch.save(dict(actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()),
            rng_after_restoration=rng,condition=condition,ordinary_start_step=65000,optimizer_start_step=0,
            feature_mean=model.feature_mean.cpu().clone(),feature_std=model.feature_std.cpu().clone(),
            source_checkpoint_sha256=request['subjects']['checkpoint']['sha256']),dest/'initialization.pt')
        write(dest/'request.json',dict(**request,condition=condition,initialization_sha256=sha(dest/'initialization.pt'),
            shared_manifest_sha256=sha(shared/'output_manifest.json'),source_receipt_sha256=identity['frozen'],clearance_sha256=identity['clearance']))
        ledger=(dest/'diagnostic_calls.jsonl').open('x',encoding='utf-8')
        state['stage']='initial_GPU32_diagnostics';model.eval()
        run_backend('initial_GPU32',model,None,data,dest,counts['diagnostics'],active,ledger,outputs)
        drift=initial_drift(outputs['initial_GPU32'],request,data)
        write(dest/'restoration.json',dict(source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],
            original1000_weights_exact=True,all_other_weights_exact=True,new323_columns_zero=True,
            original1000_normalization_exact=True,fresh_optimizer=True,optimizer_step=0,RNG_exact=True,initial_drift=drift))
        if not drift['passed']:raise ValueError('Changed input dimension initial preclamp drift exceeds1e-5; no updates.')
        if condition=='causal':
            paired_initial={}
            for corpus in CORPORA:
                other=np.load(BASE/'fit/blinded'/('initial_GPU32_'+corpus+'.npy'),mmap_mode='r',allow_pickle=False)
                own=np.asarray(outputs['initial_GPU32'][corpus])
                difference=(own.astype(np.float64)-other.astype(np.float64))*data['span'].astype(np.float64)
                maximum=float(np.abs(difference).max())
                paired_initial[corpus]=dict(max_preclip_error_rad=maximum,byte_equal=own.tobytes()==other.tobytes())
                if not np.isfinite(maximum) or maximum>1e-5:raise ValueError('Paired initial function differs; no causal updates.')
            write(dest/'initial_comparison.json',dict(compared_before_updates=True,corpora=paired_initial))
        state['stage']='training_tensor_setup'
        tensor=lambda value:torch.from_numpy(np.asarray(value)).to('cuda')
        x=tensor(data['features']);fx=tensor(data['full_state_features']);px=tensor(data['physical_features'])
        y=tensor(normalized_labels(data['target'],data['default'],data['span']))
        span64=data['span'].astype(np.float64)
        fy=tensor(data['full_state_target_change'].reshape(3057,58,2,23)/span64)
        py=tensor((data['physical_target']-data['target'][data['physical_successor']])/span64)
        cells=[tensor(ids) for ids in data['cells']];pcells=[tensor(ids) for ids in data['physical_cells']];successors=tensor(data['physical_successor'])
        parameters=list(model.parameters());model.train()
        used_centers=np.lib.format.open_memmap(dest/'used_centers.npy',mode='w+',dtype=np.int32,shape=(3000,864));used_centers[:]=-1
        used_axes=np.lib.format.open_memmap(dest/'used_axes.npy',mode='w+',dtype=np.int8,shape=(3000,864));used_axes[:]=-1
        for index in range(UPDATES):
            state.update(stage='training',attempted_global_step=65001+index,attempted_optimizer_step=index+1,
                optimizer_call_started=False,optimizer_call_returned=False,optimizer_synchronized=False,backward_returned=False)
            rows=sc[index].astype(np.int64);axes=sa[index].astype(np.int64);probes=endpoint_rows(rows,axes);mapped=data['center_map'][rows]
            active.clear();active.update(condition=condition,schedule_index=index,center_rows=rows,axes=axes,probe_rows=probes,nominal_center_indices=mapped)
            rate=cosine_rate(index)
            for group in optimizer.param_groups:group['lr']=rate
            nominal=counted_forward(model,x,'nominal',counts['training'],active,9000,44058000)
            full=counted_forward(model,fx[tensor(probes)],'full_state',counts['training'],active,9000,44058000)
            physical=counted_forward(model,px,'physical',counts['training'],active,9000,44058000)
            nl,nc=nominal_loss(nominal,y,cells)
            fl,fc=full_state_loss(nominal,full,tensor(mapped),fy[tensor(rows),tensor(axes)])
            pl,pc=physical_loss(nominal,physical,successors,py,pcells,PHYSICAL_COUNTS)
            loss=nl+COEFFICIENT*fl+pl
            active.update(nominal_loss=nl.detach(),full_state_loss=fl.detach(),physical_loss=pl.detach(),total_loss=loss.detach(),
                nominal_cell_losses=nc.detach(),full_state_cell_losses=fc.detach(),physical_cell_losses=pc.detach())
            if not all(bool(torch.isfinite(v)) for v in (nl,fl,pl,loss)):raise ValueError('Nonfinite fixed loss.')
            optimizer.zero_grad(set_to_none=True);loss.backward();state['backward_returned']=True
            grad=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True);active['preclip_gradient_norm']=grad.detach()
            state['optimizer_call_started']=True;optimizer.step();state['optimizer_call_returned']=True
            torch.cuda.synchronize();state['optimizer_synchronized']=True;state['completed_updates']=index+1
            used_centers[index]=sc[index];used_axes[index]=sa[index]
            losses.append([float(nl.detach()),float(fl.detach()),float(pl.detach()),float(loss.detach()),rate,float(grad.detach())])
            nc_records.append(nc.detach().cpu().numpy().copy());fc_records.append(fc.detach().cpu().numpy().copy());pc_records.append(pc.detach().cpu().numpy().copy())
            if (index+1)%50==0:
                used_centers.flush();used_axes.flush();save_loss_prefix(dest,losses,nc_records,fc_records,pc_records)
                atomic(dest/'progress.json',dict(condition=condition,stage='training',completed_updates=index+1,ordinary_step=65001+index,losses=losses[-1],counts=counts))
                print(condition,'update',index+1,'loss',losses[-1][:4],flush=True)
        state['stage']='save_ordinary_endpoint';used_centers.flush();used_axes.flush();save_loss_prefix(dest,losses,nc_records,fc_records,pc_records)
        exact_saved(np.asarray(used_centers),sc,'used centers');exact_saved(np.asarray(used_axes),sa,'used axes')
        if counts['training']!=expected_counter(9000,44058000):raise ValueError('Condition training budget differs.')
        if len(optimizer.state)!=6 or any(int(v['step'])!=3000 for v in optimizer.state.values()):raise ValueError('Condition AdamW endpoint differs.')
        if condition=='blinded':assert_blinded_columns(model.actor.state_dict())
        checkpoint=dict(kind='direct_absolute_native23_target',condition=condition,ordinary_final_step=68000,additional_updates=3000,optimizer_step=3000,
            fresh_optimizer=True,actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()),rng=rng_save(),
            feature_mean=torch.from_numpy(normalization['feature_mean']),feature_std=torch.from_numpy(normalization['feature_std']),
            joint_span=torch.from_numpy(data['span']),runtime_joint_span=torch.from_numpy(span64),default_q=torch.from_numpy(data['default']),
            joint_limits=torch.from_numpy(data['limits']),retained_feature_indices=torch.from_numpy(RETAINED),
            context_order='previous_action23_then_incoming_history300',context_blinded=condition=='blinded',
            full_state_coefficient=COEFFICIENT,request=request,counters=copy.deepcopy(counts),source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'])
        torch.save(checkpoint,dest/'student_head.pt');optimized=True
        write(dest/'optimization_completed.json',dict(optimization_completed=True,condition=condition,ordinary_final_step=68000,optimizer_step=3000,
            checkpoint_sha256=sha(dest/'student_head.pt'),export_validation_pending=True))
        # Training tensors are no longer needed during complete endpoint diagnostics.
        del x,fx,px,y,fy,py,cells,pcells,successors,nominal,full,physical,nl,fl,pl,loss,nc,fc,pc
        state['stage']='final_GPU32_diagnostics';model.eval()
        run_backend('final_GPU32',model,None,data,dest,counts['diagnostics'],active,ledger,outputs)
        state['stage']='promote_exact_endpoint';promoted=from_checkpoint(checkpoint).eval()
        promotion={k:dict(shape=list(v.shape),dtype=str(v.dtype),source_float32_roundtrip_exact=bool(torch.equal(v,v.to(torch.float32).to(torch.float64)))) for k,v in promoted.state_dict().items()}
        if not all(v['source_float32_roundtrip_exact'] for v in promotion.values()):raise ValueError('Nonexact promoted parameter.')
        write(dest/'promoted_parameters.json',dict(parameters=promotion,checkpoint_sha256=sha(dest/'student_head.pt')))
        export_onnx(promoted,dest/'student_head.onnx')
        state['stage']='CPU64';run_backend('CPU64',promoted,None,data,dest,counts['diagnostics'],active,ledger,outputs)
        state['stage']='GPU64';promoted=promoted.to('cuda');run_backend('GPU64',promoted,None,data,dest,counts['diagnostics'],active,ledger,outputs)
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        session=ort.InferenceSession(str(dest/'student_head.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        if session.get_providers()!=['CPUExecutionProvider']:raise ValueError('Unexpected ORT provider.')
        state['stage']='ORT64';run_backend('ORT64',None,session,data,dest,counts['diagnostics'],active,ledger,outputs)
        for label in BACKENDS:
            metrics[label]=summarize(outputs[label],data)
            metrics[label]['weighted_objective']=metrics[label]['nominal_objective']+COEFFICIENT*metrics[label]['full_state_objective']+metrics[label]['physical_objective']
            write(dest/(label+'_metrics.json'),metrics[label])
            if counts['diagnostics'][label]!=expected_counter(1437,367570):raise ValueError('Condition diagnostic budget: '+label)
        numerical=parity(outputs,data);write(dest/'export_parity.json',numerical);write(dest/'drift.json',save_drift(outputs,data,dest));export_done=True
        ledger.flush();os.fsync(ledger.fileno());ledger.close();final_identity(BASE,identity,receipt)
        write(dest/'output_manifest.json',dict(files=manifest(dest),training_request_sha256=identity['request'],frozen_receipt_sha256=identity['frozen']))
        report=dict(completed=bool(numerical['passed']),optimization_completed=True,final_export_diagnostics_completed=True,
            numerical_gate_passed=bool(numerical['passed']),export_parity_passed=bool(numerical['passed']),condition=condition,
            ordinary_final_step=68000,additional_updates=3000,optimizer_step=3000,fresh_optimizer=True,features=1323,context_features=323,
            head_output='normalized_target',fixed_full_state_coefficient=COEFFICIENT,counts=counts,metrics=metrics,
            execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',
            training_first_layer_execution='split_contiguous_1000_plus_323',export_first_layer_execution='monolithic_float64_1323',
            parity_tolerance_rad=1e-5,max_preclip_error_rad=numerical['maximum_preclamp_rad'],
            checkpoint_sha256=sha(dest/'student_head.pt'),onnx_sha256=sha(dest/'student_head.onnx'),
            normalization_sha256=sha(shared/'normalization.npz'),shared_manifest_sha256=sha(shared/'output_manifest.json'),
            training_request_sha256=identity['request'],frozen_receipt_sha256=identity['frozen'],
            output_manifest_sha256=sha(dest/'output_manifest.json'),all_frozen_inputs_unchanged=True,
            checkpoint_selection=False,native_steps=0,BFM_calls=0,elapsed_seconds=time.perf_counter()-started)
        write(dest/'report.json',report)
        if not numerical['passed']:raise ValueError('Endpoint same-weight FP64 parity failed; pair incomplete.')
        return report
    except BaseException as exc:
        errors=[]
        for value in [used_centers,used_axes]+[a for group in outputs.values() for a in group.values()]:
            if value is not None:
                try:value.flush()
                except BaseException as error:errors.append('flush: '+repr(error))
        try:
            if ledger is not None and not ledger.closed:ledger.flush();os.fsync(ledger.fileno());ledger.close()
        except BaseException as error:errors.append('ledger: '+repr(error))
        try:save_loss_prefix(dest,losses,nc_records,fc_records,pc_records)
        except BaseException as error:errors.append('loss prefix: '+repr(error))
        state['committed_loss_rows']=len(losses)
        try:state['optimizer_step_counters']=[int(optimizer.state.get(p,{}).get('step',0)) for p in model.parameters()] if optimizer is not None else []
        except BaseException as error:errors.append('optimizer counters: '+repr(error))
        try:
            evidence={};flatten_active(active,'active',evidence);np.savez_compressed(dest/'failed_active_evidence.npz',**evidence)
            if model is not None:torch.save(dict(actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()) if optimizer else None,
                state=copy.deepcopy(state),counters=counts,rng=rng_save(),resumable=False),dest/'failed_state.pt')
        except BaseException as error:errors.append('active/state preservation: '+repr(error))
        failure=dict(error=repr(exc),state=state,counts=counts,preservation_errors=errors,optimization_completed=optimized,
            final_export_diagnostics_completed=export_done,ordinary_endpoint_preserved=(dest/'student_head.pt').exists(),automatic_retry=False)
        write(dest/'failure.json',failure)
        if not (dest/'report.json').exists():write(dest/'report.json',dict(completed=False,numerical_gate_passed=False,export_parity_passed=False,**failure))
        raise

def main():
    receipt,request,identity=gate(BASE,__file__)
    dest=BASE/'fit';dest.mkdir(exist_ok=False);shared=dest/'shared';shared.mkdir()
    reports={};stage='load_shared_data'
    try:
        data=load_context_data(request,receipt['input_sha256'])
        source=torch.load(request['subjects']['checkpoint']['path'],map_location='cpu',weights_only=True)
        if source['kind']!='direct_absolute_native23_target' or source['ordinary_final_step']!=65000:raise ValueError('Exact source65000 required.')
        if source['full_state_coefficient']!=COEFFICIENT:raise ValueError('Original full-state coefficient differs.')
        original_mean=source['feature_mean'].cpu().numpy().copy();original_std=source['feature_std'].cpu().numpy().copy()
        with np.load(request['subjects']['normalization']['path'],allow_pickle=False) as norm:
            exact_saved(original_mean,norm['feature_mean'],'original mean');exact_saved(original_std,norm['feature_std'],'original std')
        if original_mean.shape!=(1000,) or original_mean.dtype!=np.float32 or original_std.shape!=(1000,) or original_std.dtype!=np.float32:raise ValueError('Original normalization schema.')
        for key,value in [('joint_span',data['span']),('runtime_joint_span',data['span'].astype(np.float64)),('default_q',data['default']),('joint_limits',data['limits']),('retained_feature_indices',RETAINED)]:exact_saved(torch.from_numpy(value),source[key],key)
        normalization=dict(feature_mean=np.r_[original_mean,data['context_mean']],feature_std=np.r_[original_std,data['context_std']],
            original_feature_mean=original_mean,original_feature_std=original_std,context_mean=data['context_mean'],context_std=data['context_std'],
            context_mean64=data['context_mean64'],context_variance64=data['context_variance64'],joint_span=data['span'],default_q=data['default'],joint_limits=data['limits'])
        np.savez_compressed(shared/'normalization.npz',**normalization)
        for key in ('nominal_context','center_context','physical_context'):np.save(shared/(key+'.npy'),data[key])
        np.savez_compressed(shared/'data_identities.npz',dataset=data['dataset'],control=data['control'],phase=data['phase'],source_frame=data['frame'],
            center_to_nominal=data['center_map'],physical_successor=data['physical_successor'],physical_dataset=data['physical_dataset'],physical_control=data['physical_control'],
            axis_group=data['full_state_centers']['axis_group'],axis_radius=data['full_state_centers']['axis_radius'])
        write(shared/'context_alignment.json',dict(**data['context_proof'],source_input_sha256=receipt['input_sha256'],
            context_sha256={key:sha(shared/(key+'.npy')) for key in ('nominal_context','center_context','physical_context')}))
        original_sc=np.load(request['schedule_paths']['centers'],allow_pickle=False);original_sa=np.load(request['schedule_paths']['axes'],allow_pickle=False)
        verify_schedule(original_sc,original_sa,data['full_state_centers']['dataset'],data['full_state_centers']['control'],data['full_state_centers']['axis_group'])
        sc,sa=fixed_schedule(original_sc,original_sa);np.save(shared/'schedule_centers.npy',sc);np.save(shared/'schedule_axes.npy',sa)
        write(shared/'runtime.json',configure_runtime(request));expanded=expand_actor(source['actor_state'])
        write(shared/'output_manifest.json',dict(files=manifest(shared),training_request_sha256=identity['request'],frozen_receipt_sha256=identity['frozen']))
        for condition in CONDITIONS:
            stage=condition
            reports[condition]=run_condition(condition,condition_data(data,condition),source,expanded,normalization,sc,sa,request,receipt,identity,shared)
            gc.collect();torch.cuda.empty_cache()
        stage='paired_saved_comparison'
        initial_comparison={};final_comparison={}
        left=torch.load(dest/'blinded/initialization.pt',map_location='cpu',weights_only=True)
        right=torch.load(dest/'causal/initialization.pt',map_location='cpu',weights_only=True)
        for key in ('actor_state','optimizer_state','rng_after_restoration','feature_mean','feature_std'):exact_saved(left[key],right[key],'paired initial '+key)
        for corpus in CORPORA:
            a=np.load(dest/'blinded'/('initial_GPU32_'+corpus+'.npy'),mmap_mode='r');b=np.load(dest/'causal'/('initial_GPU32_'+corpus+'.npy'),mmap_mode='r')
            delta=(a.astype(np.float64)-b.astype(np.float64))*data['span'].astype(np.float64)
            maximum=float(np.abs(delta).max())
            if not np.isfinite(maximum) or maximum>1e-5:raise ValueError('Paired initial function drift exceeds predeclared tolerance.')
            initial_comparison[corpus]=dict(max_preclip_error_rad=maximum,byte_equal=a.tobytes()==b.tobytes())
        for key in ('nominal_objective','full_state_objective','physical_objective','weighted_objective'):
            a=reports['blinded']['metrics']['ORT64'][key];b=reports['causal']['metrics']['ORT64'][key]
            final_comparison[key]=dict(blinded=a,causal=b,causal_minus_blinded=b-a,causal_over_blinded=None if a==0 else b/a)
        final_identity(BASE,identity,receipt)
        write(dest/'paired_report.json',dict(completed=True,conditions=list(CONDITIONS),updates_per_condition=3000,
            ordinary_final_step=68000,fixed_coefficient=COEFFICIENT,budgets=BUDGETS,
            shared_initial_actor_optimizer_RNG_normalization_exact=True,shared_schedule_exact=True,
            initial_comparison=initial_comparison,final_comparison=final_comparison,
            condition_report_sha256={key:sha(dest/key/'report.json') for key in CONDITIONS},
            input_sha256=receipt['input_sha256'],all_frozen_inputs_unchanged=True,
            checkpoint_selection=False,controller_selected=False,context_utility_not_unique_hidden_state_attribution=True))
        print('PAIR_COMPLETE',sha(dest/'paired_report.json'),flush=True)
    except BaseException as exc:
        write(dest/'paired_failure.json',dict(completed=False,stage=stage,error=repr(exc),
            finished_conditions=list(reports),condition_report_paths={key:str(dest/key/'report.json') for key in CONDITIONS if (dest/key/'report.json').exists()},
            automatic_retry=False,controller_selected=False))
        raise

if __name__=='__main__':main()
