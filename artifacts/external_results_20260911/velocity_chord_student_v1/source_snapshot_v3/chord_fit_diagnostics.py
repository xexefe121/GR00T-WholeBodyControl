"""Draft fixed initial/final prediction, complete chord metrics and ONNX accounting."""
import math
import numpy as np
import torch
import onnxruntime as ort

ROWS=3057
TOTAL=143679

def input_batch(center_features,probe_features,start,stop):
    if stop<=ROWS:return center_features[start:stop]
    flat=probe_features.reshape(-1,1069)
    if start>=ROWS:return flat[start-ROWS:stop-ROWS]
    return np.concatenate((center_features[start:],flat[:stop-ROWS]))

def evaluate_fixed(actor,center_features,probe_features,actual_seven,mean,std,span,onnx_path,ledger,destination,label):
    """562 all-input ORT batches plus one seven-input batch; no RNG calls."""
    result=np.full((TOTAL,23),np.nan,np.float32);ort_predictions=np.full_like(result,np.nan)
    actual=np.full((7,23),np.nan,np.float32);ort_actual=np.full_like(actual,np.nan)
    counts=dict(torch_rows_returned=0,onnx_rows_returned=0,torch_actual_returned=False,onnx_actual_returned=False)
    calls=0;last_ort_returned=np.empty((0,23),np.float32)
    try:
        with torch.inference_mode():
            # Preserve the original3057-row nominal forward operation for restoration.
            ledger['diagnostic_stage']=label+'_torch_centers';ledger['diagnostic_torch_rows_attempted']+=ROWS
            normalized_centers=actor(torch.from_numpy((center_features-mean)/std))
            result[:ROWS]=(normalized_centers*torch.from_numpy(span)).numpy()
            counts['torch_rows_returned']=ROWS;ledger['diagnostic_torch_rows_returned']+=ROWS
            flat=probe_features.reshape(-1,1069)
            for start in range(0,len(flat),256):
                stop=min(start+256,len(flat))
                ledger['diagnostic_stage']=label+'_torch_probes';ledger['diagnostic_batch']=[start,stop]
                ledger['diagnostic_torch_rows_attempted']+=stop-start
                x=torch.from_numpy((flat[start:stop]-mean)/std)
                result[ROWS+start:ROWS+stop]=(actor(x)*torch.from_numpy(span)).numpy()
                counts['torch_rows_returned']=ROWS+stop;ledger['diagnostic_torch_rows_returned']+=stop-start
            ledger['diagnostic_stage']=label+'_torch_actual7';ledger['diagnostic_torch_rows_attempted']+=7
            actual=(actor(torch.from_numpy((actual_seven-mean)/std))*torch.from_numpy(span)).numpy()
            counts['torch_actual_returned']=True;ledger['diagnostic_torch_rows_returned']+=7
        assert np.isfinite(result).all() and np.isfinite(actual).all()
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        ledger['diagnostic_stage']=label+'_construct_ORT'
        session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=['CPUExecutionProvider'])
        for start in range(0,TOTAL,256):
            stop=min(start+256,TOTAL)
            batch=input_batch(center_features,probe_features,start,stop)
            ledger['diagnostic_stage']=label+'_ORT_batch';ledger['diagnostic_batch']=[start,stop]
            assert ledger['head_onnx_calls_attempted']<1126
            ledger['head_onnx_calls_attempted']+=1;calls+=1
            last_ort_returned=session.run(None,{'features':batch})[0]
            ledger['head_onnx_calls_returned']+=1
            ort_predictions[start:stop]=last_ort_returned;counts['onnx_rows_returned']=stop
        ledger['diagnostic_stage']=label+'_ORT_actual7'
        assert ledger['head_onnx_calls_attempted']<1126
        ledger['head_onnx_calls_attempted']+=1;calls+=1
        last_ort_returned=session.run(None,{'features':actual_seven})[0]
        ledger['head_onnx_calls_returned']+=1
        ort_actual=last_ort_returned;counts['onnx_actual_returned']=True
    except BaseException:
        np.savez_compressed(destination/(label+'_failed_diagnostic_arrays.npz'),predicted_delta=result,
            onnx_predicted_delta=ort_predictions,actual_seven_delta=actual,actual_seven_onnx_delta=ort_actual,
            last_ort_returned=last_ort_returned,
            **{k:np.asarray(v) for k,v in counts.items()})
        ledger['diagnostic_partial_arrays']=label+'_failed_diagnostic_arrays.npz'
        ledger['diagnostic_returned_prefix']=counts
        raise
    np.savez_compressed(destination/(label+'_predictions.npz'),predicted_delta=result,
        onnx_predicted_delta=ort_predictions,actual_seven_delta=actual,actual_seven_onnx_delta=ort_actual,
        normalized_centers=normalized_centers.numpy())
    ledger['diagnostic_arrays_path']=label+'_predictions.npz'
    assert calls==563
    maximum=float(max(np.max(np.abs(result-ort_predictions)),np.max(np.abs(actual-ort_actual))))
    assert np.isfinite(ort_predictions).all() and np.isfinite(ort_actual).all() and math.isfinite(maximum)
    return dict(predicted_delta=result,onnx_predicted_delta=ort_predictions,actual_seven_delta=actual,
        actual_seven_onnx_delta=ort_actual,normalized_centers=normalized_centers.numpy(),onnx_calls=calls,onnx_max_difference=maximum,
        export_parity_passed=maximum<1e-5)

def complete_chord_metrics(predicted,centers,probe_base,probe_target,strata,span):
    center_u=centers['base_target']+predicted[:ROWS]
    probe_u=probe_base+predicted[ROWS:].reshape(ROWS,23,2,23)
    teacher_change=probe_target-centers['expert_target'][:,None,None,:]
    proposal_change=probe_u-center_u[:,None,None,:]
    lo,hi=centers['joint_limits'].T
    center_a=np.clip(center_u,lo,hi);probe_a=np.clip(probe_u,lo,hi)
    normalized=((proposal_change-teacher_change)/span)**2
    applied_difference=probe_a-center_a[:,None,None,:]-teacher_change
    cells=[]
    for dataset,group in enumerate(strata):
        for phase,ids in enumerate(group):
            cells.append(dict(dataset=dataset,phase=phase,rows=len(ids),axis_pairs=len(ids)*23,
                normalized_proposal_chord_MSE=float(normalized[ids].mean()),
                proposal_chord_error_rms_rad=float(np.sqrt(np.mean((proposal_change[ids]-teacher_change[ids])**2))),
                applied_chord_error_rms_rad=float(np.sqrt(np.mean(applied_difference[ids]**2))),
                student_clipped_probes=int(np.any(probe_u[ids]!=probe_a[ids],axis=-1).sum()),
                student_clipped_components=int(np.sum(probe_u[ids]!=probe_a[ids])),
                first24=dict(rows=len(ids[:24]),axis_pairs=len(ids[:24])*23,
                    normalized_proposal_chord_MSE=float(normalized[ids[:24]].mean()),
                    proposal_chord_error_rms_rad=float(np.sqrt(np.mean((proposal_change[ids[:24]]-teacher_change[ids[:24]])**2))),
                    applied_chord_error_rms_rad=float(np.sqrt(np.mean(applied_difference[ids[:24]]**2))),
                    student_clipped_probes=int(np.any(probe_u[ids[:24]]!=probe_a[ids[:24]],axis=-1).sum()))))
    return dict(full_nine_cell_chord_objective=float(np.mean([x['normalized_proposal_chord_MSE'] for x in cells])),
        cells=cells,all_probe_axes_and_signs_evaluated=True,checkpoint_selection=False)

def nominal_metrics(normalized,predicted,centers,strata,span,pairs,global_step):
    from fit_phase_fullbatch_once import objective
    labels=torch.from_numpy((centers['residual_rad']/span).astype(np.float32))
    loss=float(objective(torch.from_numpy(normalized),labels,strata))
    lo,hi=centers['joint_limits'].T
    proposed=centers['base_target']+predicted
    applied=np.clip(proposed,lo,hi)
    error=applied-centers['expert_target'];raw_error=predicted-centers['residual_rad']
    def summary(ids):
        return dict(rows=len(ids),applied_target_rmse_rad=float(np.sqrt(np.mean(error[ids]**2))),
            applied_target_by_joint_rmse_rad=np.sqrt(np.mean(error[ids]**2,axis=0)).tolist(),
            unclipped_target_rmse_rad=float(np.sqrt(np.mean(raw_error[ids]**2))),
            clipped_rows=int(np.any(proposed[ids]!=applied[ids],axis=-1).sum()))
    groups=[]
    for dataset,group in enumerate(strata):
        groups.append(dict(dataset=dataset,all=summary(np.concatenate(group)),
            phases=[summary(ids) for ids in group],first24_each_phase=[summary(ids[:24]) for ids in group]))
    fixed=[]
    for pair in pairs:
        i,j=pair['a_global_row'],pair['b_global_row']
        assert pair['a_control']==int(centers['control'][i]) and pair['b_control']==int(centers['control'][j])
        fixed.append(dict(pair,predicted_target_difference_rad=(applied[j]-applied[i]).tolist(),
            actual_target_difference_rad=(centers['expert_target'][j]-centers['expert_target'][i]).tolist(),
            a_target_error_rad=error[i].tolist(),b_target_error_rad=error[j].tolist(),
            predicted_residual_difference_rad=(predicted[j]-predicted[i]).tolist(),
            actual_residual_difference_rad=(centers['residual_rad'][j]-centers['residual_rad'][i]).tolist()))
    return dict(global_step=global_step,nine_cell_nominal_objective=loss,all=summary(np.arange(3057)),
        datasets=groups,query250_signed_error_rad=error[2038].tolist(),
        query250_target_rmse_rad=float(np.sqrt(np.mean(error[2038]**2))),fixed_pairs=fixed,
        predictions_reused_without_additional_inference=True)
