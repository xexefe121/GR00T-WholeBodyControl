"""Fixed complete-corpus diagnostics; output preservation precedes validation."""
from pathlib import Path
import numpy as np
import torch
from direct_contract import COUNTS,PHYSICAL_COUNTS,absolute_targets

CORPORA=('nominal','velocity','physical')
SIZES=(9904,140622,3054)
BATCH=256

def corpus_features(data):return dict(nominal=data['features'],velocity=data['velocity_features'],physical=data['physical_features'])

def run_torch_pass(model,data,destination,label,counters,active):
    """Only selected initialGPU/finalGPU/finalCPU passes call this function."""
    active.clear();active.update(stage=label,diagnostic_current_returned=False)
    outputs={};device=next(model.parameters()).device;model.eval()
    with torch.inference_mode():
        for name,size in zip(CORPORA,SIZES):
            x=corpus_features(data)[name];assert len(x)==size
            out=np.lib.format.open_memmap(Path(destination)/(label+'_'+name+'.npy'),mode='w+',dtype=np.float32,shape=(size,23));out[:]=np.nan
            active.update(stage=label,corpus=name,diagnostic_valid_rows=0,diagnostic_current_returned=False)
            try:
                for start in range(0,size,BATCH):
                    stop=min(start+BATCH,size);active.update(diagnostic_slice=[start,stop],diagnostic_input=x[start:stop].copy(),diagnostic_current_returned=False)
                    active.pop('diagnostic_output',None);counters['diagnostic_torch_rows_attempted']+=stop-start
                    returned=model(torch.from_numpy(x[start:stop]).to(device))
                    if device.type=='cuda':torch.cuda.synchronize()
                    counters['diagnostic_torch_rows_returned']+=stop-start
                    actual=returned.detach().cpu().numpy().copy();active['diagnostic_output']=actual;active['diagnostic_current_returned']=True
                    if actual.shape!=(stop-start,23) or actual.dtype!=np.float32:raise ValueError('Diagnostic Torch output schema.')
                    out[start:stop]=actual;active['diagnostic_valid_rows']=stop
                    if not np.isfinite(actual).all():raise ValueError('Nonfinite diagnostic Torch output.')
            finally:out.flush()
            outputs[name]=out
    return outputs

def run_ort_pass(onnx_path,data,destination,counters,active,session_factory=None):
    active.clear();active.update(stage='construct_final_ORT',diagnostic_current_returned=False)
    if session_factory is None:
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=['CPUExecutionProvider'])
    else:session=session_factory(onnx_path)
    outputs={}
    for name,size in zip(CORPORA,SIZES):
        x=corpus_features(data)[name];assert len(x)==size
        out=np.lib.format.open_memmap(Path(destination)/('final_ORT_'+name+'.npy'),mode='w+',dtype=np.float32,shape=(size,23));out[:]=np.nan
        active.update(stage='final_ORT',corpus=name,diagnostic_valid_rows=0,diagnostic_current_returned=False)
        try:
            for start in range(0,size,BATCH):
                stop=min(start+BATCH,size);active.update(diagnostic_slice=[start,stop],diagnostic_input=x[start:stop].copy(),diagnostic_current_returned=False)
                active.pop('diagnostic_output',None);active.pop('diagnostic_all_outputs',None)
                if counters['ORT_calls_attempted']>=601:raise ValueError('Fixed ORT call budget exhausted.')
                counters['ORT_calls_attempted']+=1
                returned=session.run(None,{'features':x[start:stop]})
                counters['ORT_calls_returned']+=1;active['diagnostic_all_outputs']=[np.asarray(v).copy() for v in returned];active['diagnostic_current_returned']=True
                actual=np.asarray(returned[0]).copy();active['diagnostic_output']=actual
                if len(returned)!=1 or actual.shape!=(stop-start,23) or actual.dtype!=np.float32:raise ValueError('ORT output schema.')
                out[start:stop]=actual;active['diagnostic_valid_rows']=stop
                if not np.isfinite(actual).all():raise ValueError('Nonfinite ORT output.')
        finally:out.flush()
        outputs[name]=out
    return outputs

def errors(pred,target,data):
    raw,applied=absolute_targets(np.asarray(pred),data['default'],data['span'],data['limits'])
    return dict(preclip_RMSE_rad=float(np.sqrt(np.mean((raw-target)**2))),applied_RMSE_rad=float(np.sqrt(np.mean((applied-target)**2))),
        preclip_per_joint_RMSE_rad=np.sqrt(np.mean((raw-target)**2,axis=0)).tolist(),
        applied_per_joint_RMSE_rad=np.sqrt(np.mean((applied-target)**2,axis=0)).tolist(),
        clipped_rows=int(np.any(raw!=applied,axis=1).sum()),clipped_components=int((raw!=applied).sum()))

def summarize(outputs,data):
    span=data['span'].astype(np.float64);nom=np.asarray(outputs['nominal']);vel=np.asarray(outputs['velocity']);physical=np.asarray(outputs['physical'])
    yn=((data['target']-data['default'])/span).astype(np.float32);nom_square=(nom-yn)**2
    center_predictions=nom[data['center_map']].astype(np.float64)
    vel_change=(data['velocity_target'].reshape(3057,23,2,23)-data['target'][data['center_map']][:,None,None])/span
    vel_difference=vel.astype(np.float64).reshape(3057,23,2,23)-center_predictions[:,None,None]-vel_change
    physical_change=(data['physical_target']-data['target'][data['physical_successor']])/span
    physical_difference=physical.astype(np.float64)-nom[data['physical_successor']].astype(np.float64)-physical_change
    nominal_cells=[];velocity_cells=[];physical_cells=[]
    for cell,ids in enumerate(data['cells']):
        first=ids[:24]
        nominal_cells.append(dict(dataset=cell//3,phase=cell%3,rows=len(ids),normalized_MSE=float(nom_square[ids].mean()),
            first24=errors(nom[first],data['target'][first],data),**errors(nom[ids],data['target'][ids],data)))
    for cell in range(9):
        ids=np.flatnonzero(np.isin(data['center_map'],data['cells'][cell]));first=ids[:24]
        flat=np.r_[tuple(np.arange(i*46,(i+1)*46) for i in ids)]
        velocity_cells.append(dict(dataset=cell//3,phase=cell%3,center_rows=len(ids),probe_rows=len(flat),response_MSE=float(np.mean(vel_difference[ids]**2)),
            first24_response_MSE=float(np.mean(vel_difference[first]**2)),**errors(vel[flat],data['velocity_target'][flat],data)))
    for cell,(ids,count) in enumerate(zip(data['physical_cells'],PHYSICAL_COUNTS)):
        first=ids[:24]
        physical_cells.append(dict(dataset=cell//3,phase=cell%3,requested=count,valid=len(ids),
            response_MSE=float(np.sum(physical_difference[ids]**2)/(count*23)),
            first24_response_MSE=float(np.mean(physical_difference[first]**2)),**errors(physical[ids],data['physical_target'][ids],data)))
    groups={}
    for name,mask in data['physical_flags'].items():
        if mask.ndim==2:mask=mask.any(axis=1)
        groups[name]=dict(rows=int(mask.sum()),response_MSE=float(np.mean(physical_difference[mask]**2)) if mask.any() else None)
    return dict(nominal_objective=float(np.mean([r['normalized_MSE'] for r in nominal_cells])),
        full_velocity_objective=float(np.mean([r['response_MSE'] for r in velocity_cells])),
        physical_objective=float(np.mean([r['response_MSE'] for r in physical_cells])),
        nominal_cells=nominal_cells,velocity_cells=velocity_cells,physical_cells=physical_cells,physical_groups=groups,
        no_checkpoint_selection=True,no_connected_stability_claim=True)

def parity(gpu,cpu,ort,data,tolerance=1e-5):
    rows={};maximum=0.;nonfinite=[]
    for name in CORPORA:
        outputs=[absolute_targets(np.asarray(values[name]),data['default'],data['span'],data['limits'])[0] for values in (gpu,cpu,ort)]
        for left,right,label in ((0,1,'GPU_CPU'),(0,2,'GPU_ORT'),(1,2,'CPU_ORT')):
            value=float(np.max(np.abs(outputs[left]-outputs[right])))
            key=name+'_'+label
            if not np.isfinite(value):rows[key]=None;nonfinite.append(key)
            else:rows[key]=value;maximum=max(maximum,value)
    return dict(passed=bool(not nonfinite and maximum<=tolerance),maximum_preclamp_rad=None if nonfinite else maximum,tolerance_rad=tolerance,
        comparisons=rows,nonfinite_comparisons=nonfinite,bitexact_claim=False,partition='each corpus independently, batch256')
