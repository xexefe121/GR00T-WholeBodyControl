"""One selected finite-feedback fit; fresh AdamW, exact55k actor, ordinary65k only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise RuntimeError('CUBLAS configuration must precede Torch import.')
import copy
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from direct_contract import RETAINED,PHYSICAL_COUNTS,normalized_labels
from direct_data import read,sha
from direct_model import DirectTarget
from direct_objective import nominal_loss,physical_loss
from full_state_contract import (SEED,UPDATES,PAIRS,ENDPOINTS,build_schedule,verify_schedule,endpoint_rows,cosine_rate,calibrate)
from full_state_objective import full_state_loss
from full_state_data import load_full_state_data
from full_state_diagnostics import BACKENDS,CORPORA,run_backend,summarize,parity,save_drift,atomic
from restoration_support import exact_saved,restore_rng
from training_support import (write,gate,final_identity,cpu_tree,rng_save,configure_runtime,
    flatten_active,counter,counted_forward,save_loss_prefix,manifest)
from promoted_model import from_checkpoint,export_onnx

BASE=Path(__file__).resolve().parent.parent

def main():
    receipt,request,identities=gate(BASE,__file__)
    dest=BASE/'fit';dest.mkdir(exist_ok=False)
    counters=dict(calibration=counter(),training=counter(),gradients=dict(attempted=0,returned=0,synchronized=0,verified=0),
        diagnostics={label:counter() for label in BACKENDS},native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
    state=dict(stage='loading_data',attempted_global_step=55000,completed_updates=0,
        optimizer_call_started=False,optimizer_call_returned=False,optimizer_synchronized=False)
    active={};model=None;optimizer=None;used_centers=None;used_axes=None;gradient_arrays={}
    losses=[];nominal_records=[];full_records=[];physical_records=[]
    outputs={label:{} for label in BACKENDS};metrics={};ledgers=[];started=time.perf_counter()
    optimization_completed=False;export_diagnostics_completed=False;coefficient=None
    try:
        data=load_full_state_data(request['paths'],request['full_state_paths'],receipt['input_sha256'])
        state['stage']='restored_data_identity'
        saved=torch.load(request['subjects']['checkpoint']['path'],map_location='cpu',weights_only=True)
        if saved['kind']!='direct_absolute_native23_target' or saved['ordinary_final_step']!=55000:raise ValueError('Exact original55k subject required.')
        mean=saved['feature_mean'].cpu().numpy().copy();std=saved['feature_std'].cpu().numpy().copy()
        if mean.shape!=(1000,) or mean.dtype!=np.float32 or std.shape!=mean.shape or std.dtype!=np.float32 or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):raise ValueError('Source normalization schema.')
        for key,value in [('joint_span',data['span']),('runtime_joint_span',data['span'].astype(np.float64)),
                          ('default_q',data['default']),('joint_limits',data['limits']),('retained_feature_indices',RETAINED)]:
            exact_saved(torch.from_numpy(value),saved[key],key)
        normalization_source=request['subjects']['normalization']['path']
        with np.load(normalization_source,allow_pickle=False) as norm:
            exact_saved(mean,norm['feature_mean'],'mean');exact_saved(std,norm['feature_std'],'std')
        shutil.copyfile(normalization_source,dest/'normalization.npz')
        labels=normalized_labels(data['target'],data['default'],data['span'])
        np.save(dest/'nominal_normalized_labels.npy',labels)
        centers=data['full_state_centers']
        np.savez_compressed(dest/'data_identities.npz',dataset=data['dataset'],control=data['control'],phase=data['phase'],source_frame=data['frame'],
            center_to_nominal=data['center_map'],physical_successor=data['physical_successor'],physical_dataset=data['physical_dataset'],physical_control=data['physical_control'],
            full_state_dataset=centers['dataset'],full_state_control=centers['control'],full_state_phase=centers['phase'],full_state_cell=centers['cell'],
            axis_group=centers['axis_group'],axis_radius=centers['axis_radius'],axis_units=centers['axis_units'])
        state['stage']='fixed_schedule_preparation'
        schedule_centers,schedule_axes,sampler_before,sampler_after=build_schedule(centers['dataset'],centers['control'],centers['axis_group'])
        verify_schedule(schedule_centers,schedule_axes,centers['dataset'],centers['control'],centers['axis_group'])
        np.save(dest/'schedule_centers.npy',schedule_centers);np.save(dest/'schedule_axes.npy',schedule_axes)
        write(dest/'sampler.json',dict(seed=SEED,kind='separate NumPy PCG64',initial_state=sampler_before,final_state=sampler_after,
            schedule_centers_sha256=sha(dest/'schedule_centers.npy'),schedule_axes_sha256=sha(dest/'schedule_axes.npy'),runtime_draws=0,
            ordering='update, dataset-phase cell, tangent group:16 centers then16 group-local axes',replacement=True))
        state['stage']='configure_deterministic_CUDA';write(dest/'runtime.json',configure_runtime(request))
        model=DirectTarget(mean,std);model.actor.load_state_dict(saved['actor_state'],strict=True);model=model.to('cuda')
        if len(list(model.parameters()))!=6 or any(parameter.dtype!=torch.float32 for parameter in model.parameters()):raise ValueError('Original float32 six-parameter actor required.')
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,foreach=False,fused=False)
        restore_rng(saved['rng']);restored_rng=rng_save()
        exact_saved(model.actor.state_dict(),saved['actor_state'],'actor');exact_saved(restored_rng,saved['rng'],'RNG')
        exact_saved(model.feature_mean.cpu(),saved['feature_mean'],'feature_mean');exact_saved(model.feature_std.cpu(),saved['feature_std'],'feature_std')
        if optimizer.state or optimizer.state_dict()['state']:raise ValueError('New objective requires empty fresh AdamW moments.')
        initialization=dict(actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()),rng_after_restoration=restored_rng,
            source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],ordinary_start_step=55000,optimizer_start_step=0,
            fresh_actor_initialization=False,fresh_optimizer=True,feature_mean=saved['feature_mean'],feature_std=saved['feature_std'])
        torch.save(initialization,dest/'initialization.pt')
        write(dest/'request.json',dict(**request,source_receipt_sha256=identities['frozen'],clearance_sha256=identities['clearance'],
            initialization_sha256=sha(dest/'initialization.pt'),normalization_sha256=sha(dest/'normalization.npz'),sampler_sha256=sha(dest/'sampler.json')))
        ledger=(dest/'diagnostic_calls.jsonl').open('x',encoding='utf-8');ledgers.append(ledger)
        state['stage']='initial_GPU32_diagnostics';model.eval()
        run_backend('initial_GPU32',model,None,data,dest,counters['diagnostics'],active,ledger,outputs)
        metrics['initial_GPU32']=summarize(outputs['initial_GPU32'],data);write(dest/'initial_GPU32_metrics.json',metrics['initial_GPU32'])
        for corpus in ('nominal','physical'):
            prior=np.load(request['restoration_predictions'][corpus],mmap_mode='r',allow_pickle=False)
            exact_saved(np.asarray(outputs['initial_GPU32'][corpus]),np.asarray(prior),'initial GPU32 identical partition '+corpus)
        prior_velocity=np.load(request['restoration_predictions']['velocity'],mmap_mode='r',allow_pickle=False)
        overlap=np.asarray(outputs['initial_GPU32']['full_state']).reshape(3057,58,2,23)[:,35:58].reshape(140622,23)
        delta=(overlap.astype(np.float64)-prior_velocity.astype(np.float64))*data['span'].astype(np.float64)
        restoration=dict(source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],actor_exact=True,normalization_exact=True,RNG_exact=True,
            fresh_optimizer=True,optimizer_state_count=0,optimizer_step=0,nominal_and_physical_predictions_exact=True,
            overlap_partition_changed=True,overlap_byte_gate_required=False,overlap_max_preclip_difference_rad=float(np.max(np.abs(delta))),
            overlap_RMS_preclip_difference_rad=float(np.sqrt(np.mean(delta**2))),all_restoration_checks_passed=True)
        write(dest/'restoration.json',restoration);del overlap,delta,prior_velocity,prior
        state['stage']='training_tensor_setup'
        tensor=lambda value:torch.from_numpy(np.asarray(value)).to('cuda')
        x=tensor(data['features']);y=tensor(labels);fx=tensor(data['full_state_features']);px=tensor(data['physical_features'])
        span64=data['span'].astype(np.float64)
        fy=tensor(data['full_state_target_change'].reshape(3057,58,2,23)/span64)
        py=tensor((data['physical_target']-data['target'][data['physical_successor']])/span64)
        cells=[tensor(ids) for ids in data['cells']];pcells=[tensor(ids) for ids in data['physical_cells']];successors=tensor(data['physical_successor'])

        def batch_losses(index,kind):
            rows=schedule_centers[index].astype(np.int64);axes=schedule_axes[index].astype(np.int64);probes=endpoint_rows(rows,axes);mapped=data['center_map'][rows]
            active.clear();active.update(stage=kind,schedule_index=index,center_rows=rows.copy(),axes=axes.copy(),probe_rows=probes.copy(),nominal_center_indices=mapped.copy())
            cap_calls,cap_rows=(3,14686) if kind=='calibration' else (30000,146860000)
            nominal=counted_forward(model,x,'nominal',counters[kind],active,cap_calls,cap_rows)
            full=counted_forward(model,fx[tensor(probes)],'full_state',counters[kind],active,cap_calls,cap_rows)
            physical=counted_forward(model,px,'physical',counters[kind],active,cap_calls,cap_rows)
            nl,nc=nominal_loss(nominal,y,cells)
            fl,fc=full_state_loss(nominal,full,tensor(mapped),fy[tensor(rows),tensor(axes)])
            pl,pc=physical_loss(nominal,physical,successors,py,pcells,PHYSICAL_COUNTS)
            active.update(nominal_loss=nl.detach(),full_state_loss=fl.detach(),physical_loss=pl.detach(),
                nominal_cell_losses=nc.detach(),full_state_cell_losses=fc.detach(),physical_cell_losses=pc.detach())
            if not all(bool(torch.isfinite(value)) for value in (nl,fl,pl)):raise ValueError('Nonfinite objective component.')
            return (nl,fl,pl),(nc,fc,pc)

        state['stage']='one_initial_gradient_calibration';model.train()
        calibration_losses,calibration_cells=batch_losses(0,'calibration')
        np.savez_compressed(dest/'calibration_forward_outputs.npz',
            nominal=active['nominal_prediction'].cpu().numpy().copy(),
            full_state=active['full_state_prediction'].cpu().numpy().copy(),
            physical=active['physical_prediction'].cpu().numpy().copy(),
            nominal_cell_losses=calibration_cells[0].detach().cpu().numpy().copy(),
            full_state_cell_losses=calibration_cells[1].detach().cpu().numpy().copy(),
            physical_cell_losses=calibration_cells[2].detach().cpu().numpy().copy())
        parameters=list(model.parameters());gradient_pieces={}
        for index,(name,value) in enumerate(zip(('nominal','full_state','physical'),calibration_losses)):
            active['current_gradient']=dict(name=name,returned=False,synchronized=False,verified=False)
            counters['gradients']['attempted']+=1
            returned=torch.autograd.grad(value,parameters,retain_graph=index<2,create_graph=False,allow_unused=False)
            counters['gradients']['returned']+=1;active['current_gradient']['returned']=True;active[name+'_gradients']=returned
            torch.cuda.synchronize();counters['gradients']['synchronized']+=1;active['current_gradient']['synchronized']=True
            gradient_pieces[name]=[value.detach().cpu().numpy().copy() for value in returned]
            for parameter_index,array in enumerate(gradient_pieces[name]):gradient_arrays[name+'_'+str(parameter_index)]=array
            np.savez_compressed(dest/'calibration_gradients.npz',**gradient_arrays)
            if len(returned)!=6 or any(value.shape!=parameter.shape or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()) for value,parameter in zip(returned,parameters)):raise ValueError('Returned calibration gradient schema/nonfinite.')
            counters['gradients']['verified']+=1;active['current_gradient']['verified']=True
        calibration=calibrate(gradient_pieces);coefficient=calibration['coefficient']
        exact_saved(model.actor.state_dict(),saved['actor_state'],'actor after calibration');exact_saved(rng_save(),restored_rng,'RNG after calibration')
        if optimizer.state or any(parameter.grad is not None for parameter in parameters):raise ValueError('Calibration altered optimizer or accumulated gradients.')
        calibration.update(losses={name:float(value.detach()) for name,value in zip(('nominal','full_state','physical'),calibration_losses)},
            schedule_index=0,gradient_sha256=sha(dest/'calibration_gradients.npz'),parameter_names=[name for name,_ in model.actor.named_parameters()],
            forward_outputs_sha256=sha(dest/'calibration_forward_outputs.npz'),
            model_unchanged=True,RNG_unchanged=True,optimizer_updates=0,counters=copy.deepcopy(counters['gradients']))
        write(dest/'coefficient.json',calibration)
        del calibration_losses,calibration_cells,returned,gradient_pieces
        used_centers=np.lib.format.open_memmap(dest/'used_centers.npy',mode='w+',dtype=np.int32,shape=(UPDATES,PAIRS));used_centers[:]=-1
        used_axes=np.lib.format.open_memmap(dest/'used_axes.npy',mode='w+',dtype=np.int8,shape=(UPDATES,PAIRS));used_axes[:]=-1
        state['stage']='fixed_10000_updates'
        for index in range(UPDATES):
            state.update(attempted_global_step=55001+index,attempted_optimizer_step=index+1,
                optimizer_call_started=False,optimizer_call_returned=False,optimizer_synchronized=False,backward_returned=False)
            rate=cosine_rate(index)
            for group in optimizer.param_groups:group['lr']=rate
            (nl,fl,pl),(nc,fc,pc)=batch_losses(index,'training');loss=nl+coefficient*fl+pl
            active['total_loss']=loss.detach()
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite weighted loss.')
            optimizer.zero_grad(set_to_none=True);state['backward_returned']=False;loss.backward();state['backward_returned']=True
            grad=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True);active['preclip_gradient_norm']=grad.detach()
            state['optimizer_call_started']=True;optimizer.step();state['optimizer_call_returned']=True
            torch.cuda.synchronize();state['optimizer_synchronized']=True;state['completed_updates']=index+1
            used_centers[index]=schedule_centers[index];used_axes[index]=schedule_axes[index]
            losses.append([float(nl.detach()),float(fl.detach()),float(pl.detach()),float(loss.detach()),rate,float(grad.detach())])
            nominal_records.append(nc.detach().cpu().numpy().copy());full_records.append(fc.detach().cpu().numpy().copy());physical_records.append(pc.detach().cpu().numpy().copy())
            if (index+1)%50==0:
                used_centers.flush();used_axes.flush();save_loss_prefix(dest,losses,nominal_records,full_records,physical_records)
                atomic(dest/'progress.json',dict(stage='training',completed_updates=index+1,ordinary_step=55001+index,
                    coefficient=coefficient,losses=losses[-1],counters=counters,elapsed_seconds=time.perf_counter()-started))
                print('update',index+1,'global',55001+index,'losses',losses[-1][:4],flush=True)
        state['stage']='save_ordinary_final';used_centers.flush();used_axes.flush();save_loss_prefix(dest,losses,nominal_records,full_records,physical_records)
        exact_saved(np.asarray(used_centers),schedule_centers,'executed centers');exact_saved(np.asarray(used_axes),schedule_axes,'executed axes')
        if len(optimizer.state)!=6 or any(int(value['step'])!=10000 for value in optimizer.state.values()):raise ValueError('Final fresh AdamW counters differ.')
        if counters['training']!=dict(calls_attempted=30000,calls_returned=30000,calls_synchronized=30000,calls_verified=30000,rows_attempted=146860000,rows_returned=146860000,rows_verified=146860000):raise ValueError('Training budget mismatch.')
        checkpoint=dict(kind='direct_absolute_native23_target',ordinary_final_step=65000,additional_updates=10000,optimizer_step=10000,
            fresh_optimizer=True,seed=SEED,actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()),rng=rng_save(),
            feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),joint_span=torch.from_numpy(data['span']),
            runtime_joint_span=torch.from_numpy(span64),default_q=torch.from_numpy(data['default']),joint_limits=torch.from_numpy(data['limits']),
            retained_feature_indices=torch.from_numpy(RETAINED),output='normalized_absolute_target',request=request,counters=copy.deepcopy(counters),
            full_state_coefficient=coefficient,coefficient_sha256=sha(dest/'coefficient.json'),source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'])
        torch.save(checkpoint,dest/'student_head.pt');optimization_completed=True
        write(dest/'optimization_completed.json',dict(optimization_completed=True,ordinary_final_step=65000,optimizer_step=10000,
            checkpoint_sha256=sha(dest/'student_head.pt'),coefficient_sha256=sha(dest/'coefficient.json'),export_validation_pending=True))
        state['stage']='final_GPU32_diagnostics';model.eval()
        run_backend('final_GPU32',model,None,data,dest,counters['diagnostics'],active,ledger,outputs)
        state['stage']='promote_exact_final_weights';promoted=from_checkpoint(checkpoint).eval()
        promotion={name:dict(shape=list(value.shape),dtype=str(value.dtype),source_float32_roundtrip_exact=bool(torch.equal(value,value.to(torch.float32).to(torch.float64)))) for name,value in promoted.state_dict().items()}
        if not all(value['source_float32_roundtrip_exact'] for value in promotion.values()):raise ValueError('Nonexact parameter promotion.')
        write(dest/'promoted_parameters.json',dict(checkpoint_sha256=sha(dest/'student_head.pt'),parameters=promotion,execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32'))
        state['stage']='manual_FP64_export';export_onnx(promoted,dest/'student_head.onnx')
        state['stage']='final_CPU64_diagnostics';run_backend('CPU64',promoted,None,data,dest,counters['diagnostics'],active,ledger,outputs)
        state['stage']='final_GPU64_diagnostics';promoted=promoted.to('cuda')
        run_backend('GPU64',promoted,None,data,dest,counters['diagnostics'],active,ledger,outputs)
        state['stage']='final_ORT64_diagnostics'
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        session=ort.InferenceSession(str(dest/'student_head.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        if session.get_providers()!=['CPUExecutionProvider']:raise ValueError('Unexpected ORT provider.')
        run_backend('ORT64',None,session,data,dest,counters['diagnostics'],active,ledger,outputs)
        for label in BACKENDS:
            metrics[label]=summarize(outputs[label],data)
            metrics[label]['weighted_objective']=metrics[label]['nominal_objective']+coefficient*metrics[label]['full_state_objective']+metrics[label]['physical_objective']
            write(dest/(label+'_metrics.json'),metrics[label])
        numerical=parity(outputs,data);write(dest/'export_parity.json',numerical)
        write(dest/'drift.json',save_drift(outputs,data,dest));export_diagnostics_completed=True
        for label in BACKENDS:
            if counters['diagnostics'][label]!=dict(calls_attempted=1437,calls_returned=1437,calls_synchronized=1437,calls_verified=1437,rows_attempted=367570,rows_returned=367570,rows_verified=367570):raise ValueError('Diagnostic budget mismatch: '+label)
        if counters['calibration']!=dict(calls_attempted=3,calls_returned=3,calls_synchronized=3,calls_verified=3,rows_attempted=14686,rows_returned=14686,rows_verified=14686) or counters['gradients']!=dict(attempted=3,returned=3,synchronized=3,verified=3):raise ValueError('Calibration budget mismatch.')
        for file in ledgers:file.flush();os.fsync(file.fileno());file.close()
        state['stage']='final_rehash';final_identity(BASE,identities,receipt)
        write(dest/'output_manifest.json',dict(files=manifest(dest),training_request_sha256=identities['request'],frozen_receipt_sha256=identities['frozen']))
        report=dict(completed=bool(numerical['passed']),optimization_completed=True,final_export_diagnostics_completed=True,
            numerical_gate_passed=bool(numerical['passed']),export_parity_passed=bool(numerical['passed']),ordinary_final_step=65000,
            additional_updates=10000,optimizer_step=10000,fresh_optimizer=True,restored_start_step=55000,features=1000,head_output='normalized_target',
            nominal_rows=9904,full_state_endpoint_rows=354612,physical_rows=3054,full_state_cells=54,full_state_coefficient=coefficient,
            coefficient_sha256=sha(dest/'coefficient.json'),restoration_sha256=sha(dest/'restoration.json'),
            execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',
            ELU_implementation='Where(x>0,x,Exp(Min(x,0))-1)',parity_tolerance_rad=1e-5,max_preclip_error_rad=numerical['maximum_preclamp_rad'],
            metrics=metrics,export_parity=numerical,counters=counters,checkpoint_sha256=sha(dest/'student_head.pt'),onnx_sha256=sha(dest/'student_head.onnx'),
            normalization_sha256=sha(dest/'normalization.npz'),training_request_sha256=identities['request'],frozen_receipt_sha256=identities['frozen'],
            output_manifest_sha256=sha(dest/'output_manifest.json'),all_frozen_inputs_unchanged=True,checkpoint_selection=False,
            BFM_calls=0,native_steps=0,FP32_ONNX_release=False,hardware_authorized=False,elapsed_seconds=time.perf_counter()-started)
        write(dest/'report.json',report)
        if not numerical['passed']:raise ValueError('Final same-weight FP64 CPU/GPU/ORT preclamp parity failed; no release.')
        state['stage']='COMPLETE';atomic(dest/'progress.json',dict(stage='COMPLETE',completed_updates=10000,ordinary_step=65000,optimizer_step=10000,counters=counters))
        print('COMPLETE',report['checkpoint_sha256'],report['onnx_sha256'],flush=True)
    except BaseException as exc:
        errors=[]
        for value in [used_centers,used_axes]+[array for groups in outputs.values() for array in groups.values()]:
            if value is not None:
                try:value.flush()
                except BaseException as error:errors.append('array flush: '+repr(error))
        for file in ledgers:
            try:
                if not file.closed:file.flush();os.fsync(file.fileno());file.close()
            except BaseException as error:errors.append('ledger flush: '+repr(error))
        try:save_loss_prefix(dest,losses,nominal_records,full_records,physical_records)
        except BaseException as error:errors.append('loss prefix: '+repr(error))
        state['committed_loss_rows']=len(losses)
        try:state['optimizer_step_counters']=[int(optimizer.state.get(parameter,{}).get('step',0)) for parameter in model.parameters()] if optimizer is not None else []
        except BaseException as error:errors.append('optimizer counters: '+repr(error))
        try:
            evidence={};flatten_active(active,'active',evidence);np.savez_compressed(dest/'failed_active_evidence.npz',**evidence)
            if gradient_arrays:np.savez_compressed(dest/'calibration_gradients_partial.npz',**gradient_arrays)
        except BaseException as error:errors.append('active arrays: '+repr(error))
        try:
            if model is not None:torch.save(dict(actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
                state=copy.deepcopy(state),counters=copy.deepcopy(counters),rng=rng_save(),resumable=False),dest/'failed_state.pt')
        except BaseException as error:errors.append('failed checkpoint: '+repr(error))
        failure=dict(error=repr(exc),state=state,counters=counters,preservation_errors=errors,optimization_completed=optimization_completed,
            final_export_diagnostics_completed=export_diagnostics_completed,ordinary_final_preserved=(dest/'student_head.pt').exists(),automatic_retry_allowed=False)
        write(dest/'failure.json',failure)
        if not (dest/'report.json').exists():write(dest/'report.json',dict(completed=False,numerical_gate_passed=False,export_parity_passed=False,**failure))
        raise

if __name__=='__main__':main()
