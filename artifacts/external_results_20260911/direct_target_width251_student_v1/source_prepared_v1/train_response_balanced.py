"""One reviewed width512 warm continuation; unchanged teacher-energy response objective."""
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
from balanced_response_objective import balanced_full_state_loss
from response_contract import UPDATES,COEFFICIENT,BUDGETS,cosine_rate,START_STEP,FINAL_STEP,OPTIMIZER_START,OPTIMIZER_FINAL
from response_data import load_response_data
from width512 import WiderContextTarget,expand_source,validate_source
from context_diagnostics import BACKENDS,CORPORA,run_backend,summarize,parity,save_drift,atomic
from width512_promoted import from_checkpoint,export_onnx
from response_support import gate,final_response_identity
from width_restore import verify_width_restoration
from response_diagnostics import summarize_balanced,save_balanced_prefix
from balance_contract import GROUP_WEIGHTS,WEIGHT_RULE
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
        if prior.shape!=actual.shape or prior.dtype!=np.float32:raise ValueError('Saved causal71000 prediction schema.')
        delta=(actual.astype(np.float64)-prior.astype(np.float64))*data['span'].astype(np.float64)
        value=float(np.max(np.abs(delta)))
        if not np.isfinite(value):raise ValueError('Nonfinite restored initial drift.')
        maximum=max(maximum,value)
        result[corpus]=dict(max_preclip_error_rad=value,RMS_preclip_error_rad=float(np.sqrt(np.mean(delta**2))),
            byte_equal=actual.tobytes()==prior.tobytes())
    return dict(passed=maximum<=1e-5,tolerance_rad=1e-5,max_preclip_error_rad=maximum,corpora=result,
        source_ordinary_step=71000,changed_MatMul_dimension=True,stored_first_layer_width=1323,stored_hidden_width=512,old_block_contractions_preserved=True,first_layer_execution='split_old256_new256_original1000_plus323',byte_gate_required=False)

def run_continuation(data,source,normalization,sc,sa,request,receipt,identity,shared):
    condition='causal';dest=BASE/'fit'
    counts=dict(training=counter(),diagnostics={name:counter() for name in BACKENDS},calibration_forward_calls=0,
        calibration_gradient_calls=0,BFM_calls=0,native_calls=0,manual_export_trace_calls=0)
    state=dict(condition=condition,stage='initializing',attempted_global_step=START_STEP,completed_updates=0,
        optimizer_call_started=False,optimizer_call_returned=False,optimizer_synchronized=False)
    active={};model=None;optimizer=None;used_centers=None;used_axes=None;ledger=None
    losses=[];nc_records=[];fc_records=[];pc_records=[];original_losses=[];weighted_cell_records=[];outputs={name:{} for name in BACKENDS};metrics={}
    optimized=False;export_done=False;started=time.perf_counter()
    try:
        expansion=expand_source(source)
        model=WiderContextTarget(normalization['feature_mean'],normalization['feature_std'])
        model.actor.load_state_dict(expansion['actor_state'],strict=True);model=model.to('cuda')
        if len(list(model.parameters()))!=6 or any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Six float32 parameter tensors required.')
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-5,foreach=False,fused=False)
        optimizer.load_state_dict(copy.deepcopy(expansion['optimizer_state']))
        restore_rng(source['rng']);rng=rng_save()
        restoration=verify_width_restoration(model,optimizer,source,expansion,rng)
        torch.save(dict(actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()),
            rng_after_restoration=rng,expansion_seed=expansion['expansion_seed'],expansion_generator_state=expansion['expansion_generator_state'],
            condition=condition,ordinary_start_step=START_STEP,optimizer_start_step=OPTIMIZER_START,hidden_width=512,
            feature_mean=model.feature_mean.cpu().clone(),feature_std=model.feature_std.cpu().clone(),
            source_checkpoint_sha256=request['subjects']['checkpoint']['sha256']),dest/'initialization.pt')
        write(dest/'request.json',dict(request,condition=condition,initialization_sha256=sha(dest/'initialization.pt'),
            shared_manifest_sha256=sha(shared/'output_manifest.json'),source_receipt_sha256=identity['frozen'],clearance_sha256=identity['clearance']))
        ledger=(dest/'diagnostic_calls.jsonl').open('x',encoding='utf-8')
        state['stage']='initial_GPU32_diagnostics';model.eval()
        run_backend('initial_GPU32',model,None,data,dest,counts['diagnostics'],active,ledger,outputs)
        drift=initial_drift(outputs['initial_GPU32'],request,data)
        write(dest/'restoration.json',dict(source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],
            **restoration,initial_drift=drift))
        if not drift['passed']:raise ValueError('Restored initial preclamp drift exceeds1e-5; no updates.')
        state['stage']='training_tensor_setup'
        tensor=lambda value:torch.from_numpy(np.asarray(value)).to('cuda')
        x=tensor(data['features']);fx=tensor(data['full_state_features']);px=tensor(data['physical_features'])
        y=tensor(normalized_labels(data['target'],data['default'],data['span']))
        span64=data['span'].astype(np.float64)
        fy=tensor(data['full_state_target_change'].reshape(3057,58,2,23)/span64)
        py=tensor((data['physical_target']-data['target'][data['physical_successor']])/span64)
        cells=[tensor(ids) for ids in data['cells']];pcells=[tensor(ids) for ids in data['physical_cells']];successors=tensor(data['physical_successor'])
        parameters=list(model.parameters());model.train()
        used_centers=np.lib.format.open_memmap(dest/'used_centers.npy',mode='w+',dtype=np.int32,shape=(UPDATES,864));used_centers[:]=-1
        used_axes=np.lib.format.open_memmap(dest/'used_axes.npy',mode='w+',dtype=np.int8,shape=(UPDATES,864));used_axes[:]=-1
        for index in range(UPDATES):
            state.update(stage='training',attempted_global_step=START_STEP+1+index,attempted_optimizer_step=OPTIMIZER_START+1+index,
                optimizer_call_started=False,optimizer_call_returned=False,optimizer_synchronized=False,backward_returned=False)
            rows=sc[index].astype(np.int64);axes=sa[index].astype(np.int64);probes=endpoint_rows(rows,axes);mapped=data['center_map'][rows]
            active.clear();active.update(condition=condition,schedule_index=index,center_rows=rows,axes=axes,probe_rows=probes,nominal_center_indices=mapped)
            rate=cosine_rate(index)
            for group in optimizer.param_groups:group['lr']=rate
            nominal=counted_forward(model,x,'nominal',counts['training'],active,BUDGETS['training_forward_calls'],BUDGETS['training_forward_rows'])
            full=counted_forward(model,fx[tensor(probes)],'full_state',counts['training'],active,BUDGETS['training_forward_calls'],BUDGETS['training_forward_rows'])
            physical=counted_forward(model,px,'physical',counts['training'],active,BUDGETS['training_forward_calls'],BUDGETS['training_forward_rows'])
            nl,nc=nominal_loss(nominal,y,cells)
            balanced=balanced_full_state_loss(nominal,full,tensor(mapped),fy[tensor(rows),tensor(axes)])
            fl=balanced.weighted_objective;fc=balanced.original_cells
            original_fl=balanced.original_objective;weighted_fc=balanced.weighted_cells
            pl,pc=physical_loss(nominal,physical,successors,py,pcells,PHYSICAL_COUNTS)
            loss=nl+COEFFICIENT*fl+pl
            active.update(nominal_loss=nl.detach(),balanced_full_state_loss=fl.detach(),physical_loss=pl.detach(),balanced_total_loss=loss.detach(),
                nominal_cell_losses=nc.detach(),full_state_cell_losses=fc.detach(),physical_cell_losses=pc.detach(),
                original_full_state_loss=original_fl.detach(),balanced_full_state_cell_losses=weighted_fc.detach())
            if not all(bool(torch.isfinite(v)) for v in (nl,fl,pl,loss)):raise ValueError('Nonfinite fixed loss.')
            optimizer.zero_grad(set_to_none=True);loss.backward();state['backward_returned']=True
            grad=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True);active['preclip_gradient_norm']=grad.detach()
            state['optimizer_call_started']=True;optimizer.step();state['optimizer_call_returned']=True
            torch.cuda.synchronize();state['optimizer_synchronized']=True;state['completed_updates']=index+1
            used_centers[index]=sc[index];used_axes[index]=sa[index]
            losses.append([float(nl.detach()),float(fl.detach()),float(pl.detach()),float(loss.detach()),rate,float(grad.detach())])
            nc_records.append(nc.detach().cpu().numpy().copy());fc_records.append(fc.detach().cpu().numpy().copy());pc_records.append(pc.detach().cpu().numpy().copy())
            original_losses.append([float(original_fl.detach()),float((nl+COEFFICIENT*original_fl+pl).detach())])
            weighted_cell_records.append(weighted_fc.detach().cpu().numpy().copy())
            if (index+1)%50==0:
                used_centers.flush();used_axes.flush();save_balanced_prefix(dest,losses,nc_records,fc_records,pc_records,original_losses,weighted_cell_records)
                atomic(dest/'progress.json',dict(condition=condition,stage='training',completed_updates=index+1,ordinary_step=START_STEP+1+index,losses=losses[-1],counts=counts))
                print(condition,'update',index+1,'loss',losses[-1][:4],flush=True)
        state['stage']='save_ordinary_endpoint';used_centers.flush();used_axes.flush();save_balanced_prefix(dest,losses,nc_records,fc_records,pc_records,original_losses,weighted_cell_records)
        exact_saved(np.asarray(used_centers),sc,'used centers');exact_saved(np.asarray(used_axes),sa,'used axes')
        if counts['training']!=expected_counter(BUDGETS['training_forward_calls'],BUDGETS['training_forward_rows']):raise ValueError('Condition training budget differs.')
        if len(optimizer.state)!=6 or any(int(v['step'])!=OPTIMIZER_FINAL for v in optimizer.state.values()):raise ValueError('Condition AdamW endpoint differs.')
        checkpoint=dict(kind='direct_absolute_native23_target',condition=condition,ordinary_final_step=FINAL_STEP,additional_updates=UPDATES,optimizer_step=OPTIMIZER_FINAL,
            fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],expansion_seed=expansion['expansion_seed'],
            expansion_generator_state=expansion['expansion_generator_state'],actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()),rng=rng_save(),
            feature_mean=torch.from_numpy(normalization['feature_mean']),feature_std=torch.from_numpy(normalization['feature_std']),
            joint_span=torch.from_numpy(data['span']),runtime_joint_span=torch.from_numpy(span64),default_q=torch.from_numpy(data['default']),
            joint_limits=torch.from_numpy(data['limits']),retained_feature_indices=torch.from_numpy(RETAINED),
            context_order='previous_action23_then_incoming_history300',context_blinded=False,
            full_state_coefficient=COEFFICIENT,response_group_weights=GROUP_WEIGHTS,response_weight_rule=WEIGHT_RULE,request=request,counters=copy.deepcopy(counts),source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'])
        torch.save(checkpoint,dest/'student_head.pt');optimized=True
        write(dest/'optimization_completed.json',dict(optimization_completed=True,condition=condition,ordinary_final_step=FINAL_STEP,optimizer_step=OPTIMIZER_FINAL,
            checkpoint_sha256=sha(dest/'student_head.pt'),export_validation_pending=True))
        # Training tensors are no longer needed during complete endpoint diagnostics.
        del x,fx,px,y,fy,py,cells,pcells,successors,nominal,full,physical,nl,fl,pl,loss,nc,fc,pc,balanced,original_fl,weighted_fc
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
            metrics[label]=summarize_balanced(outputs[label],data)
            write(dest/(label+'_metrics.json'),metrics[label])
            if counts['diagnostics'][label]!=expected_counter(1437,367570):raise ValueError('Condition diagnostic budget: '+label)
        numerical=parity(outputs,data);write(dest/'export_parity.json',numerical);write(dest/'drift.json',save_drift(outputs,data,dest));export_done=True
        ledger.flush();os.fsync(ledger.fileno());ledger.close();final_response_identity(BASE,identity,receipt)
        write(dest/'output_manifest.json',dict(files=manifest(dest),training_request_sha256=identity['request'],frozen_receipt_sha256=identity['frozen']))
        report=dict(completed=bool(numerical['passed']),optimization_completed=True,final_export_diagnostics_completed=True,
            numerical_gate_passed=bool(numerical['passed']),export_parity_passed=bool(numerical['passed']),condition=condition,
            ordinary_final_step=FINAL_STEP,additional_updates=UPDATES,optimizer_step=OPTIMIZER_FINAL,fresh_optimizer=False,features=1323,context_features=323,
            head_output='normalized_target',fixed_full_state_coefficient=COEFFICIENT,counts=counts,metrics=metrics,
            response_group_weights=GROUP_WEIGHTS,response_weight_rule=WEIGHT_RULE,
            ordinary_start_step=START_STEP,optimizer_start_step=OPTIMIZER_START,context_and_normalization_reused=True,
            architecture=[1323,512,512,23],hidden_width=512,expansion_seed=expansion['expansion_seed'],
            initial_restoration=drift,training_loss_columns=['nominal','balanced_full_state','physical','balanced_total','learning_rate','preclip_gradient_norm'],
            source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],
            energy_source_sha256=request['subjects']['energy_source']['sha256'],
            direct_subject_sha256={key:value['sha256'] for key,value in request['subjects'].items()},
            execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',
            training_first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',
            parity_tolerance_rad=1e-5,max_preclip_error_rad=numerical['maximum_preclamp_rad'],
            checkpoint_sha256=sha(dest/'student_head.pt'),onnx_sha256=sha(dest/'student_head.onnx'),
            normalization_sha256=sha(shared/'normalization.npz'),shared_manifest_sha256=sha(shared/'output_manifest.json'),
            training_request_sha256=identity['request'],frozen_receipt_sha256=identity['frozen'],
            output_manifest_sha256=sha(dest/'output_manifest.json'),all_frozen_inputs_unchanged=True,
            checkpoint_selection=False,native_steps=0,BFM_calls=0,elapsed_seconds=time.perf_counter()-started)
        write(dest/'report.json',report)
        if not numerical['passed']:raise ValueError('Endpoint same-weight FP64 parity failed; continuation incomplete.')
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
        try:save_balanced_prefix(dest,losses,nc_records,fc_records,pc_records,original_losses,weighted_cell_records)
        except BaseException as error:errors.append('loss prefix: '+repr(error))
        state['committed_loss_rows']=len(losses)
        try:state['optimizer_step_counters']=[int(optimizer.state.get(p,{}).get('step',0)) for p in model.parameters()] if optimizer is not None else []
        except BaseException as error:errors.append('optimizer counters: '+repr(error))
        try:
            evidence={};flatten_active(active,'active',evidence);np.savez_compressed(dest/'failed_active_evidence.npz',**evidence)
            if model is not None:torch.save(dict(actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()) if optimizer else None,
                state=copy.deepcopy(state),counters=counts,rng=rng_save(),resumable=False),dest/'failed_state.pt')
        except BaseException as error:errors.append('active/state preservation: '+repr(error))
        final_check=False
        try:final_response_identity(BASE,identity,receipt);final_check=True
        except BaseException as error:errors.append('final identity: '+repr(error))
        failure=dict(error=repr(exc),state=state,counts=counts,preservation_errors=errors,optimization_completed=optimized,
            all_frozen_inputs_unchanged=final_check,
            final_export_diagnostics_completed=export_done,ordinary_endpoint_preserved=(dest/'student_head.pt').exists(),automatic_retry=False)
        write(dest/'failure.json',failure)
        if not (dest/'report.json').exists():write(dest/'report.json',dict(completed=False,numerical_gate_passed=False,export_parity_passed=False,**failure))
        raise

def main():
    import shutil
    receipt,request,identity=gate(BASE,__file__)
    dest=BASE/'fit';dest.mkdir(exist_ok=False);shared=dest/'shared';shared.mkdir()
    stage='load_reused_data';run_started=False
    try:
        data,normalization,sc,sa=load_response_data(request,receipt['input_sha256'])
        source=torch.load(request['subjects']['checkpoint']['path'],map_location='cpu',weights_only=True)
        validate_source(source)
        for key in ('feature_mean','feature_std'):
            exact_saved(source[key],torch.from_numpy(normalization[key]),'saved full1323 '+key)
        for key,value in [('joint_span',data['span']),('runtime_joint_span',data['span'].astype(np.float64)),('default_q',data['default']),('joint_limits',data['limits']),('retained_feature_indices',RETAINED)]:
            exact_saved(torch.from_numpy(value),source[key],key)
        # Preserve the qualified files byte-for-byte; no context normalization or schedule generation.
        copied={}
        for key,path in request['context_paths'].items():
            name='source_output_manifest.json' if key=='manifest' else ('source_'+Path(path).name if key.startswith('schedule_') else Path(path).name)
            shutil.copyfile(path,shared/name);copied[name]=sha(shared/name)
            if copied[name]!=sha(path):raise ValueError('Copied reused context bytes differ: '+key)
        for key,path in request['schedule_paths'].items():
            shutil.copyfile(path,shared/Path(path).name)
            if sha(shared/Path(path).name)!=sha(path):raise ValueError('Copied full schedule differs: '+key)
        write(shared/'schedule_lineage.json',data['schedule_lineage'])
        write(shared/'reused_inputs.json',dict(context_paths=request['context_paths'],schedule_paths=request['schedule_paths'],copied_sha256=copied,
            chronology_proof_sha256=request['subjects']['context_proof']['sha256'],
            context_reconstructed=False,normalization_recomputed=False,schedule_generated=False))
        write(shared/'runtime.json',configure_runtime(request))
        write(shared/'output_manifest.json',dict(files=manifest(shared),training_request_sha256=identity['request'],frozen_receipt_sha256=identity['frozen']))
        stage='run_continuation';run_started=True
        report=run_continuation(data,source,normalization,sc,sa,request,receipt,identity,shared)
        print('CONTINUATION_COMPLETE',sha(dest/'report.json'),flush=True)
    except BaseException as exc:
        if not run_started:
            unchanged=False;identity_error=None
            try:final_response_identity(BASE,identity,receipt);unchanged=True
            except BaseException as error:identity_error=repr(error)
            failure=dict(completed=False,optimization_completed=False,stage=stage,error=repr(exc),
                task_forward_calls=0,optimizer_updates=0,all_frozen_inputs_unchanged=unchanged,
                final_identity_error=identity_error,automatic_retry=False)
            write(dest/'setup_failure.json',failure);write(dest/'report.json',failure)
        raise

if __name__=='__main__':main()
