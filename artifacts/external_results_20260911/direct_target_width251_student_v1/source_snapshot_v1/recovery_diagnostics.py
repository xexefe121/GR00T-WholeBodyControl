"""Fixed complete-corpus diagnostics and same-weight64-bit export checks."""
from pathlib import Path
import json
import os
import time
import numpy as np
import torch
from direct_contract import absolute_targets, normalized_labels, PHYSICAL_COUNTS

CORPORA = ('nominal','full_state','physical','recovery')
SIZES = (9904,354612,3054,1018)
FEATURE_KEYS = ('features','full_state_features','physical_features','recovery_features')
BACKENDS = ('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')
BATCH = 256
CALLS = 1441

def atomic(path, value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    for attempt in range(41):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt==40:raise
            time.sleep(.05)

def run_backend(label, model, session, data, dest, counters, active, ledger, outputs):
    if label not in BACKENDS:raise ValueError('Unexpected diagnostic backend.')
    counts=counters[label]
    for corpus,size,key in zip(CORPORA,SIZES,FEATURE_KEYS):
        features=data[key]
        if features.shape!=(size,1323) or features.dtype!=np.float32:raise ValueError('Fixed corpus input schema.')
        out=np.lib.format.open_memmap(Path(dest)/(label+'_'+corpus+'.npy'),mode='w+',dtype=np.float32,shape=(size,23))
        out[:]=np.nan;outputs[label][corpus]=out
        try:
            for start in range(0,size,BATCH):
                stop=min(start+BATCH,size);rows=stop-start
                active.clear();active.update(stage='diagnostic',backend=label,corpus=corpus,start=start,stop=stop,
                    features=features[start:stop].copy(),returned=False,synchronized=False,verified=False)
                if counts['calls_attempted']>=CALLS:raise ValueError('Fixed per-backend diagnostic budget exhausted.')
                counts['calls_attempted']+=1;counts['rows_attempted']+=rows
                try:
                    if label=='ORT64':
                        returned=session.run(None,{'features':features[start:stop]})
                        counts['calls_returned']+=1;counts['rows_returned']+=rows;active['returned']=True
                        active['all_outputs']=[np.asarray(value).copy() for value in returned]
                        counts['calls_synchronized']+=1;active['synchronized']=True
                        if len(returned)!=1:raise ValueError('ORT output count.')
                        actual=np.asarray(returned[0]).copy()
                    else:
                        device='cpu' if label=='CPU64' else 'cuda'
                        tensor=torch.from_numpy(features[start:stop]).to(device)
                        with torch.inference_mode():returned=model(tensor)
                        counts['calls_returned']+=1;counts['rows_returned']+=rows;active['returned']=True
                        active['returned_tensor']=returned
                        if device=='cuda':torch.cuda.synchronize()
                        counts['calls_synchronized']+=1;active['synchronized']=True
                        actual=returned.detach().cpu().numpy().copy()
                    active['output']=actual
                    if actual.shape!=(rows,23) or actual.dtype!=np.float32:raise ValueError('Public diagnostic output schema.')
                    out[start:stop]=actual
                    if not np.isfinite(actual).all():raise ValueError('Nonfinite returned diagnostic output.')
                    counts['calls_verified']+=1;counts['rows_verified']+=rows;active['verified']=True
                finally:
                    ledger.write(json.dumps({key:active[key] for key in ('backend','corpus','start','stop','returned','synchronized','verified')})+'\n')
                    ledger.flush()
                if counts['calls_attempted']%25==0:
                    out.flush();atomic(Path(dest)/'progress.json',dict(stage='diagnostic',backend=label,corpus=corpus,stop=stop,counts=counters))
        finally:out.flush();ledger.flush();os.fsync(ledger.fileno())

def target_errors(prediction,target,data):
    raw,applied=absolute_targets(np.asarray(prediction),data['default'],data['span'],data['limits'])
    return dict(preclip_RMSE_rad=float(np.sqrt(np.mean((raw-target)**2))),
        applied_RMSE_rad=float(np.sqrt(np.mean((applied-target)**2))),
        preclip_per_joint_RMSE_rad=np.sqrt(np.mean((raw-target)**2,axis=0)).tolist(),
        applied_per_joint_RMSE_rad=np.sqrt(np.mean((applied-target)**2,axis=0)).tolist(),
        clipped_rows=int(np.any(raw!=applied,axis=1).sum()),clipped_components=int((raw!=applied).sum()))

def summarize(outputs,data):
    nominal=np.asarray(outputs['nominal']);full=np.asarray(outputs['full_state']);physical=np.asarray(outputs['physical'])
    labels=normalized_labels(data['target'],data['default'],data['span'])
    nominal_square=(nominal-labels)**2
    full_change=data['full_state_target_change'].reshape(3057,58,2,23)/data['span'].astype(np.float64)
    full_error=(full.astype(np.float64).reshape(3057,58,2,23)
        -nominal[data['center_map']].astype(np.float64)[:,None,None,:]-full_change)
    odd_error=(full_error[:,:,1]-full_error[:,:,0])*.5
    even_error=(full_error[:,:,1]+full_error[:,:,0])*.5
    physical_change=(data['physical_target']-data['target'][data['physical_successor']])/data['span'].astype(np.float64)
    physical_error=physical.astype(np.float64)-nominal[data['physical_successor']].astype(np.float64)-physical_change
    nominal_cells=[];full_cells=[];physical_cells=[]
    for cell,ids in enumerate(data['cells']):
        nominal_cells.append(dict(dataset=cell//3,phase=cell%3,rows=len(ids),normalized_MSE=float(nominal_square[ids].mean()),
            first24=target_errors(nominal[ids[:24]],data['target'][ids[:24]],data),
            **target_errors(nominal[ids],data['target'][ids],data)))
    for cell,centers in enumerate(data['full_state_cells']):
        for group,axes in enumerate(data['full_state_group_axes']):
            selected=full_error[centers[:,None],axes[None,:]]
            first=full_error[centers[:24,None],axes[None,:]]
            rows=(((centers[:,None]*58+axes[None,:])[:,:,None]*2)+np.arange(2)[None,None,:]).reshape(-1)
            full_cells.append(dict(dataset=cell//3,phase=cell%3,tangent_group=group,
                center_rows=len(centers),axes=len(axes),endpoint_rows=len(rows),
                response_MSE=float(np.mean(selected**2)),first24_response_MSE=float(np.mean(first**2)),
                odd_response_MSE=float(np.mean(odd_error[centers[:,None],axes[None,:]]**2)),
                even_response_MSE=float(np.mean(even_error[centers[:,None],axes[None,:]]**2)),
                zero_response_MSE=float(np.mean(full_change[centers[:,None],axes[None,:]]**2)),
                **target_errors(full[rows],data['full_state_target'][rows],data)))
    for cell,(ids,count) in enumerate(zip(data['physical_cells'],PHYSICAL_COUNTS)):
        physical_cells.append(dict(dataset=cell//3,phase=cell%3,requested=count,valid=len(ids),
            response_MSE=float(np.sum(physical_error[ids]**2)/(count*23)),
            first24_response_MSE=float(np.mean(physical_error[ids[:24]]**2)),
            **target_errors(physical[ids],data['physical_target'][ids],data)))
    flags={}
    for key,array in data['full_state_flags'].items():
        mask=np.any(array,axis=-1)
        flags[key]=dict(endpoint_rows=int(mask.sum()),raw_row_weighted_response_MSE=float(np.mean(full_error[mask]**2)) if mask.any() else None)
    return dict(nominal_objective=float(np.mean([cell['normalized_MSE'] for cell in nominal_cells])),
        full_state_objective=float(np.mean([cell['response_MSE'] for cell in full_cells])),
        full_state_odd_MSE=float(np.mean([cell['odd_response_MSE'] for cell in full_cells])),
        full_state_even_MSE=float(np.mean([cell['even_response_MSE'] for cell in full_cells])),
        full_state_zero_response_MSE=float(np.mean([cell['zero_response_MSE'] for cell in full_cells])),
        full_state_group_comparison=[dict(tangent_group=g,
            response_MSE=float(np.mean([cell['response_MSE'] for cell in full_cells if cell['tangent_group']==g])),
            zero_response_MSE=float(np.mean([cell['zero_response_MSE'] for cell in full_cells if cell['tangent_group']==g]))) for g in range(6)],
        physical_objective=float(np.mean([cell['response_MSE'] for cell in physical_cells])),
        nominal_cells=nominal_cells,full_state_cells=full_cells,physical_cells=physical_cells,
        full_state_flag_diagnostics=flags,full_state_radius_division=False,no_connected_stability_claim=True)

def parity(outputs,data,tolerance=1e-5):
    comparisons={};maximum=0.;bad=[]
    for corpus in CORPORA:
        raw={label:absolute_targets(np.asarray(outputs[label][corpus]),data['default'],data['span'],data['limits'])[0]
             for label in ('CPU64','GPU64','ORT64')}
        for left,right in (('CPU64','GPU64'),('CPU64','ORT64'),('GPU64','ORT64')):
            key=corpus+'_'+left+'_'+right;error=float(np.max(np.abs(raw[left]-raw[right])))
            if not np.isfinite(error):bad.append(key);comparisons[key]=None
            else:comparisons[key]=error;maximum=max(maximum,error)
    return dict(passed=not bad and maximum<=tolerance,tolerance_rad=tolerance,
        maximum_preclamp_rad=None if bad else maximum,comparisons=comparisons,nonfinite_comparisons=bad,
        public_dtype='float32',internal_dtype='float64',FP32_export_release=False)

def save_drift(outputs,data,dest):
    results={}
    for backend in ('CPU64','GPU64','ORT64'):
        for corpus in CORPORA:
            before,before_applied=absolute_targets(np.asarray(outputs['final_GPU32'][corpus]),data['default'],data['span'],data['limits'])
            after,after_applied=absolute_targets(np.asarray(outputs[backend][corpus]),data['default'],data['span'],data['limits'])
            drift=after-before;key=backend+'_vs_final_GPU32_'+corpus
            np.save(Path(dest)/('drift_'+key+'.npy'),drift)
            old_clip=before!=before_applied;new_clip=after!=after_applied;changed=old_clip!=new_clip
            results[key]=dict(shape=list(drift.shape),max_abs_preclamp_rad=float(np.max(np.abs(drift))),
                RMS_preclamp_rad=float(np.sqrt(np.mean(drift**2))),old_clipped_rows=int(np.any(old_clip,axis=1).sum()),
                old_clipped_components=int(old_clip.sum()),new_clipped_rows=int(np.any(new_clip,axis=1).sum()),
                new_clipped_components=int(new_clip.sum()),clipping_changed_rows=int(np.any(changed,axis=1).sum()),
                clipping_changed_components=int(changed.sum()))
    return results
