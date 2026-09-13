"""One cleared fresh direct-target fit; no automatic retry or physical calls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise RuntimeError('Deterministic CUBLAS configuration required before Torch import.')
import copy
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from direct_contract import RETAINED,PHYSICAL_COUNTS,SEED,UPDATES,weighted_normalization,normalized_labels,cosine_rate
from direct_data import read,sha,load_data
from direct_model import DirectTarget,export_onnx
from direct_objective import nominal_loss,velocity_loss,physical_loss
from direct_diagnostics import run_torch_pass,run_ort_pass,summarize,parity

BASE=Path(__file__).resolve().parent.parent

def write(path,value):
    with Path(path).open('w',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def atomic_write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    for attempt in range(41):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt==40:raise
            time.sleep(.05)

def frozen_check(receipt):
    for path,digest in receipt['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed frozen input: '+path)
    for name,digest in receipt['source_sha256'].items():
        if sha(Path(receipt['source_directory'])/name)!=digest:raise ValueError('Changed source: '+name)

def gate():
    frozen_path=BASE/'training_frozen_inputs.json';request_path=BASE/'training_request.json';clearance_path=BASE/'training_clearance.json'
    receipt=read(frozen_path);request=read(request_path);clearance=read(clearance_path)
    if Path(receipt['source_directory']).resolve()!=Path(__file__).parent.resolve():raise ValueError('Executing outside the frozen source directory.')
    if sha(request_path)!=receipt['training_request_sha256']:raise ValueError('Request identity differs.')
    frozen_check(receipt)
    if not clearance['approved'] or clearance['frozen_receipt_sha256']!=sha(frozen_path) or clearance['request_sha256']!=sha(request_path):raise ValueError('Missing concrete fit clearance.')
    if sha(clearance['review_path'])!=clearance['review_sha256']:raise ValueError('Final launch review changed.')
    review=read(clearance['review_path'])
    if review[clearance['review_pass_field']] is not True:raise ValueError('Final launch review not passed.')
    if review['training_request_sha256']!=sha(request_path) or review['frozen_receipt_sha256']!=sha(frozen_path):raise ValueError('Review does not bind actual request/receipt.')
    if sha(clearance['launcher_path'])!=clearance['launcher_sha256']:raise ValueError('Launcher changed.')
    if request['kind']!='one_fresh_direct_absolute_target_fit' or not request['root_selected'] or request['updates']!=5000 or request['seed']!=SEED:raise ValueError('Unexpected selected fit.')
    expected=dict(training_head_rows=70550000,diagnostic_torch_rows=460740,ORT_calls=601,native_steps=0,BFM_calls=0)
    if request['budgets']!=expected:raise ValueError('Unexpected fixed budgets.')
    runtime=request['runtime']
    if Path(sys.executable).resolve()!=Path(runtime['python_path']).resolve():raise ValueError('Wrong isolated Python runtime.')
    if sha(runtime['verification_path'])!=runtime['verification_sha256'] or read(runtime['verification_path'])[runtime['pass_field']] is not True:raise ValueError('Runtime verification missing.')
    return receipt,request,dict(request=sha(request_path),frozen=sha(frozen_path),clearance=sha(clearance_path))

def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def rng_save():
    state=np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state().clone(),torch_cuda=[v.cpu().clone() for v in torch.cuda.get_rng_state_all()],
        numpy=dict(name=state[0],keys=state[1].tolist(),position=state[2],has_gauss=state[3],cached_gaussian=state[4]),python=random.getstate())

def configure_runtime(request):
    runtime=request['runtime'];torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if torch.__version__!=runtime['torch_version'] or torch.version.cuda!=runtime['cuda_version']:raise ValueError('Torch/CUDA version differs.')
    if not torch.cuda.is_available():raise ValueError('Selected CUDA runtime unavailable; no CPU fallback.')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
    return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_TF32=torch.backends.cudnn.allow_tf32,AMP=False,parameter_dtype='float32',optimizer_foreach=False,optimizer_fused=False)

def flatten_active(value,prefix,result):
    if torch.is_tensor(value):result[prefix]=value.detach().cpu().numpy().copy()
    elif isinstance(value,np.ndarray):result[prefix]=value.copy()
    elif isinstance(value,dict):
        for key,item in value.items():flatten_active(item,prefix+'_'+str(key),result)
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):flatten_active(item,prefix+'_'+str(index),result)
    elif value is not None:result[prefix]=np.asarray(value)

def save_loss_prefix(dest,losses,nominal_cells,velocity_cells,physical_cells):
    np.save(dest/'training_progress.npy',np.asarray(losses,dtype=np.float64).reshape(-1,6))
    np.save(dest/'nominal_cell_losses.npy',np.asarray(nominal_cells,dtype=np.float64).reshape(-1,15))
    np.save(dest/'velocity_cell_losses.npy',np.asarray(velocity_cells,dtype=np.float64).reshape(-1,9))
    np.save(dest/'physical_cell_losses.npy',np.asarray(physical_cells,dtype=np.float64).reshape(-1,9))

def main():
    receipt,request,identities=gate()
    dest=BASE/'fit';dest.mkdir(exist_ok=False)
    counters=dict(training_head_rows_attempted=0,training_head_rows_returned=0,diagnostic_torch_rows_attempted=0,diagnostic_torch_rows_returned=0,ORT_calls_attempted=0,ORT_calls_returned=0)
    state=dict(stage='initializing',attempted_step=0,completed_updates=0,optimizer_call_started=False,optimizer_call_returned=False,optimizer_synchronized=False)
    active={};model=None;optimizer=None;data=None;used_rows=None;used_axes=None
    losses=[];nominal_records=[];velocity_records=[];physical_records=[];started=time.perf_counter()
    optimization_completed=False;export_diagnostics_completed=False
    try:
        state['stage']='loading_bound_data';data=load_data(request['paths'],receipt['input_sha256'])
        mean,std,mean64,variance64=weighted_normalization(data['features'],data['cells'])
        labels=normalized_labels(data['target'],data['default'],data['span'])
        np.savez_compressed(dest/'normalization.npz',feature_mean=mean,feature_std=std,weighted_mean64=mean64,weighted_variance64=variance64,
            retained_feature_indices=RETAINED,joint_span=data['span'],runtime_joint_span=data['span'].astype(np.float64),default_q=data['default'],joint_limits=data['limits'])
        np.savez_compressed(dest/'data_identities.npz',dataset=data['dataset'],control=data['control'],phase=data['phase'],source_frame=data['frame'],
            center_to_nominal=data['center_map'],physical_successor=data['physical_successor'],physical_dataset=data['physical_dataset'],physical_control=data['physical_control'])
        np.save(dest/'nominal_normalized_labels.npy',labels)
        state['stage']='configure_deterministic_CUDA';runtime=configure_runtime(request);write(dest/'runtime.json',runtime)
        rng_before=rng_save();model=DirectTarget(mean,std)
        if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Nonfloat32 parameter.')
        rng_after=rng_save()
        torch.save(dict(seed=SEED,actor_state=cpu_tree(model.actor.state_dict()),rng_before_initialization=rng_before,rng_after_initialization=rng_after,
            default_torch_initialization=True,zeroed_last_layer=False,optimizer_updates=0),dest/'initialization.pt')
        model=model.to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5,foreach=False,fused=False)
        write(dest/'request.json',dict(**request,source_receipt_sha256=identities['frozen'],clearance_sha256=identities['clearance'],
            normalization_sha256=sha(dest/'normalization.npz'),initialization_sha256=sha(dest/'initialization.pt')))
        state['stage']='initial_GPU_diagnostics'
        initial=run_torch_pass(model,data,dest,'initial_GPU',counters,active);initial_metrics=summarize(initial,data);write(dest/'initial_metrics.json',initial_metrics)
        state['stage']='training_tensor_setup'
        tensor=lambda value:torch.from_numpy(np.asarray(value)).to('cuda')
        x=tensor(data['features']);y=tensor(labels);vx=tensor(data['velocity_features']);px=tensor(data['physical_features'])
        span64=data['span'].astype(np.float64)
        velocity_change=(data['velocity_target'].reshape(3057,23,2,23)-data['target'][data['center_map']][:,None,None])/span64
        physical_change=(data['physical_target']-data['target'][data['physical_successor']])/span64
        vy=tensor(velocity_change);py=tensor(physical_change)
        cells=[tensor(ids) for ids in data['cells']];pcells=[tensor(ids) for ids in data['physical_cells']];successors=tensor(data['physical_successor'])
        used_rows=np.lib.format.open_memmap(dest/'replayed_center_rows.npy',mode='w+',dtype=np.int32,shape=(5000,576));used_rows[:]=-1
        used_axes=np.lib.format.open_memmap(dest/'replayed_axes.npy',mode='w+',dtype=np.int8,shape=(5000,576));used_axes[:]=-1
        model.train();state['stage']='fixed_5000_updates'
        for index in range(5000):
            state.update(attempted_step=index+1,optimizer_call_started=False,optimizer_call_returned=False,optimizer_synchronized=False)
            rows=data['sampled_rows'][index].astype(np.int64);axes=data['sampled_axes'][index].astype(np.int64)
            probe_rows=((rows*23+axes)[:,None]*2+np.arange(2)[None]).reshape(-1)
            mapped=data['center_map'][rows];active.clear();active.update(step=index+1,center_rows=rows.copy(),axes=axes.copy(),probe_rows=probe_rows.copy(),nominal_center_indices=mapped.copy())
            rate=cosine_rate(index)
            for group in optimizer.param_groups:group['lr']=rate
            state.update(forward_stage='nominal',forward_call_returned=False,forward_synchronized=False);counters['training_head_rows_attempted']+=9904
            nominal=model(x);state['forward_call_returned']=True;active['nominal_prediction']=nominal.detach()
            torch.cuda.synchronize();state['forward_synchronized']=True;counters['training_head_rows_returned']+=9904
            state.update(forward_stage='velocity',forward_call_returned=False,forward_synchronized=False);counters['training_head_rows_attempted']+=1152
            probes=model(vx[tensor(probe_rows)]);state['forward_call_returned']=True;active['velocity_prediction']=probes.detach()
            torch.cuda.synchronize();state['forward_synchronized']=True;counters['training_head_rows_returned']+=1152
            state.update(forward_stage='physical',forward_call_returned=False,forward_synchronized=False);counters['training_head_rows_attempted']+=3054
            physical=model(px);state['forward_call_returned']=True;active['physical_prediction']=physical.detach()
            torch.cuda.synchronize();state['forward_synchronized']=True;counters['training_head_rows_returned']+=3054
            nl,nc=nominal_loss(nominal,y,cells)
            vl,vc=velocity_loss(nominal,probes,tensor(mapped),vy[tensor(rows),tensor(axes)])
            pl,pc=physical_loss(nominal,physical,successors,py,pcells,PHYSICAL_COUNTS)
            loss=nl+vl+pl;active.update(nominal_loss=nl.detach(),velocity_loss=vl.detach(),physical_loss=pl.detach(),total_loss=loss.detach(),
                nominal_cell_losses=nc.detach(),velocity_cell_losses=vc.detach(),physical_cell_losses=pc.detach())
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss.')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);active['preclip_gradient_norm']=grad.detach()
            state['optimizer_call_started']=True;optimizer.step();state['optimizer_call_returned']=True
            torch.cuda.synchronize();state['optimizer_synchronized']=True;state['completed_updates']=index+1
            # Record the completed update's exact schedule only after synchronization.
            used_rows[index]=data['sampled_rows'][index];used_axes[index]=data['sampled_axes'][index]
            losses.append([float(nl.detach()),float(vl.detach()),float(pl.detach()),float(loss.detach()),rate,float(grad.detach())])
            nominal_records.append(nc.detach().cpu().numpy().copy());velocity_records.append(vc.detach().cpu().numpy().copy());physical_records.append(pc.detach().cpu().numpy().copy())
            if (index+1)%50==0:
                used_rows.flush();used_axes.flush();save_loss_prefix(dest,losses,nominal_records,velocity_records,physical_records)
                atomic_write(dest/'progress.json',dict(completed_updates=index+1,requested_updates=5000,losses=losses[-1],counters=counters,elapsed_seconds=time.perf_counter()-started))
                print(json.dumps(dict(step=index+1,nominal=losses[-1][0],velocity=losses[-1][1],physical=losses[-1][2])),flush=True)
        state['stage']='save_ordinary_final';used_rows.flush();used_axes.flush();save_loss_prefix(dest,losses,nominal_records,velocity_records,physical_records)
        if counters['training_head_rows_returned']!=70550000 or counters['training_head_rows_attempted']!=70550000:raise ValueError('Training row budget mismatch.')
        if not np.array_equal(used_rows,data['sampled_rows']) or not np.array_equal(used_axes,data['sampled_axes']):raise ValueError('Schedule replay mismatch.')
        checkpoint=dict(kind='direct_absolute_native23_target',ordinary_final_step=5000,additional_updates=5000,seed=SEED,
            actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()),rng=rng_save(),
            feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),joint_span=torch.from_numpy(data['span']),
            runtime_joint_span=torch.from_numpy(span64),default_q=torch.from_numpy(data['default']),joint_limits=torch.from_numpy(data['limits']),
            retained_feature_indices=torch.from_numpy(RETAINED),output='normalized_absolute_target',request=request,counters=copy.deepcopy(counters))
        torch.save(checkpoint,dest/'student_head.pt');optimization_completed=True
        write(dest/'optimization_completed.json',dict(optimization_completed=True,ordinary_final_step=5000,checkpoint_sha256=sha(dest/'student_head.pt'),export_validation_pending=True))
        state['stage']='final_export';export_onnx(model,dest/'student_head.onnx')
        state['stage']='final_GPU_diagnostics';final_gpu=run_torch_pass(model,data,dest,'final_GPU',counters,active)
        state['stage']='final_CPU_diagnostics';cpu_model=copy.deepcopy(model).cpu();final_cpu=run_torch_pass(cpu_model,data,dest,'final_CPU',counters,active)
        state['stage']='final_ORT_diagnostics';final_ort=run_ort_pass(dest/'student_head.onnx',data,dest,counters,active)
        final_metrics=summarize(final_gpu,data);write(dest/'final_metrics.json',final_metrics)
        numerical=parity(final_gpu,final_cpu,final_ort,data);write(dest/'export_parity.json',numerical)
        export_diagnostics_completed=True
        if counters['diagnostic_torch_rows_attempted']!=460740 or counters['diagnostic_torch_rows_returned']!=460740 or counters['ORT_calls_attempted']!=601 or counters['ORT_calls_returned']!=601:raise ValueError('Diagnostic budget mismatch.')
        state['stage']='final_rehash'
        for name,path in [('request',BASE/'training_request.json'),('frozen',BASE/'training_frozen_inputs.json'),('clearance',BASE/'training_clearance.json')]:
            if sha(path)!=identities[name]:raise ValueError('Final identity changed: '+name)
        frozen_check(receipt)
        report=dict(completed=bool(numerical['passed']),optimization_completed=True,final_export_diagnostics_completed=True,
            numerical_gate_passed=bool(numerical['passed']),export_parity_passed=bool(numerical['passed']),ordinary_final_step=5000,additional_updates=5000,
            features=1000,head_output='normalized_target',
            fresh_initialization=True,seed=SEED,nominal_rows=9904,velocity_probe_rows=140622,physical_rows=3054,
            initial_metrics=initial_metrics,final_metrics=final_metrics,export_parity=numerical,counters=counters,
            checkpoint_sha256=sha(dest/'student_head.pt'),onnx_sha256=sha(dest/'student_head.onnx'),all_frozen_inputs_unchanged=True,
            BFM_calls=0,native_steps=0,checkpoint_selection=False,hardware_authorized=False,elapsed_seconds=time.perf_counter()-started)
        write(dest/'report.json',report)
        if not numerical['passed']:raise ValueError('Final CPU/GPU/ORT parity failed; ordinary final retained.')
        state['stage']='COMPLETE';atomic_write(dest/'progress.json',dict(completed_updates=5000,stage='COMPLETE',counters=counters));print(json.dumps(dict(completed=True,checkpoint_sha256=report['checkpoint_sha256'],onnx_sha256=report['onnx_sha256'])),flush=True)
    except BaseException as exc:
        preservation=[]
        for value in (used_rows,used_axes):
            if value is not None:
                try:value.flush()
                except BaseException as error:preservation.append('memmap flush: '+repr(error))
        try:save_loss_prefix(dest,losses,nominal_records,velocity_records,physical_records)
        except BaseException as error:preservation.append('loss prefix: '+repr(error))
        state['committed_loss_rows']=len(losses)
        try:
            state['optimizer_step_counters']=[int(optimizer.state.get(p,{}).get('step',0)) for p in model.parameters()] if optimizer is not None else []
        except BaseException as error:preservation.append('optimizer counters unavailable: '+repr(error))
        try:
            evidence={};flatten_active(active,'active',evidence);np.savez_compressed(dest/'failed_active_evidence.npz',**evidence)
        except BaseException as error:preservation.append('active arrays: '+repr(error))
        try:
            if model is not None:torch.save(dict(actor_state=cpu_tree(model.actor.state_dict()),optimizer_state=cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
                state=copy.deepcopy(state),counters=copy.deepcopy(counters),rng=rng_save(),resumable=False),dest/'failed_state.pt')
        except BaseException as error:preservation.append('failed checkpoint: '+repr(error))
        write(dest/'failure.json',dict(error=repr(exc),state=state,counters=counters,preservation_errors=preservation,
            optimization_completed=optimization_completed,final_export_diagnostics_completed=export_diagnostics_completed,
            ordinary_final_preserved=(dest/'student_head.pt').exists(),automatic_retry_allowed=False))
        if not (dest/'report.json').exists():write(dest/'report.json',dict(completed=False,optimization_completed=optimization_completed,
            final_export_diagnostics_completed=export_diagnostics_completed,numerical_gate_passed=False,export_parity_passed=False,
            error=repr(exc),state=state,counters=counters,ordinary_final_preserved=(dest/'student_head.pt').exists()))
        raise

if __name__=='__main__':main()
