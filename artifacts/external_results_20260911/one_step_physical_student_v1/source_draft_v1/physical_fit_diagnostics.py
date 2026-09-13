"""Original diagnostics plus fixed physical rows; all attempted calls preserved."""
import numpy as np
import torch
from chord_fit_diagnostics import evaluate_fixed as evaluate_legacy
from physical_training_objective import REQUESTED_COUNTS

def sync_counts(ledger):
    old = ledger['legacy_diagnostics']
    physical = ledger['physical_diagnostics']
    for suffix in ('attempted','returned'):
        ledger['head_onnx_calls_'+suffix] = old['head_onnx_calls_'+suffix]+physical['head_onnx_calls_'+suffix]
        ledger['diagnostic_torch_rows_'+suffix] = old['diagnostic_torch_rows_'+suffix]+physical['diagnostic_torch_rows_'+suffix]

def evaluate_physical(actor, features, mean, std, span, onnx_path, ledger, destination,
                      label, expected_calls, session_factory=None):
    """No BFM/native calls; batch256 fixed; failed returned data remain available."""
    count = len(features)
    predicted = np.full((count,23),np.nan,np.float32)
    exported = np.full_like(predicted,np.nan)
    valid_torch = np.zeros(count,bool);valid_onnx = np.zeros(count,bool)
    current_input = np.empty((0,1069),np.float32)
    last_output = np.empty((0,23),np.float32)
    last_outputs = {}
    try:
        with torch.inference_mode():
            for start in range(0,count,256):
                stop=min(start+256,count);ledger['stage']=label+'_physical_Torch';ledger['batch']=[start,stop]
                current_input=features[start:stop].copy()
                ledger['diagnostic_torch_rows_attempted']+=stop-start
                returned=actor(torch.from_numpy((current_input-mean)/std))*torch.from_numpy(span)
                ledger['diagnostic_torch_rows_returned']+=stop-start
                last_output=returned.numpy().copy()
                predicted[start:stop]=last_output;valid_torch[start:stop]=True
                assert np.isfinite(last_output).all() and last_output.shape==(stop-start,23)
        if session_factory is None:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
            options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=['CPUExecutionProvider'])
        else:session=session_factory(onnx_path)
        for start in range(0,count,256):
            stop=min(start+256,count);ledger['stage']=label+'_physical_ORT';ledger['batch']=[start,stop]
            current_input=features[start:stop].copy()
            assert ledger['head_onnx_calls_attempted']<expected_calls
            ledger['head_onnx_calls_attempted']+=1
            returned=session.run(None,{'features':current_input})
            ledger['head_onnx_calls_returned']+=1
            last_outputs={'last_returned_output_'+str(i):np.asarray(value).copy() for i,value in enumerate(returned)}
            last_output=np.asarray(returned[0]).copy()
            assert len(returned)==1 and last_output.shape==(stop-start,23) and last_output.dtype==np.float32
            exported[start:stop]=last_output;valid_onnx[start:stop]=True
            assert np.isfinite(last_output).all()
        assert valid_torch.all() and valid_onnx.all()
        maximum=float(np.max(np.abs(predicted-exported))) if count else 0.
        assert np.isfinite(maximum)
        np.savez_compressed(destination/(label+'_physical_predictions.npz'),
            predicted_delta=predicted,onnx_predicted_delta=exported)
        return dict(predicted_delta=predicted,onnx_predicted_delta=exported,
            onnx_max_difference=maximum,export_parity_passed=maximum<1e-5,
            onnx_calls=(count+255)//256)
    except BaseException:
        np.savez_compressed(destination/(label+'_failed_physical_diagnostics.npz'),
            predicted_delta=predicted,onnx_predicted_delta=exported,
            valid_torch=valid_torch,valid_onnx=valid_onnx,
            current_input=current_input,last_returned_output=last_output,**last_outputs)
        raise

def evaluate_all(actor, center_features, probe_features, actual_seven, mean, std, span,
                 onnx_path, ledger, destination, label, physical):
    """Separate old1126 counter preserves the original function's exact hard budget."""
    try:
        original=evaluate_legacy(actor,center_features,probe_features,actual_seven,mean,std,span,
            onnx_path,ledger['legacy_diagnostics'],destination,label)
        added=evaluate_physical(actor,physical['features'],mean,std,span,onnx_path,
            ledger['physical_diagnostics'],destination,label,physical['budgets']['physical_head_onnx_calls'])
        original['physical']=added
        original['onnx_calls']+=added['onnx_calls']
        original['onnx_max_difference']=max(original['onnx_max_difference'],added['onnx_max_difference'])
        original['export_parity_passed']=original['export_parity_passed'] and added['export_parity_passed']
        return original
    finally:sync_counts(ledger)

def physical_metrics(center_delta, branch_delta, centers, physical, span):
    successors=physical['successor']
    center_proposal=centers['base_target'][successors]+center_delta[successors]
    proposal=physical['base']+branch_delta
    teacher_change=physical['target']-centers['expert_target'][successors]
    pair=((proposal-center_proposal)-teacher_change)/span
    absolute=(proposal-physical['target'])/span
    lo,hi=centers['joint_limits'].T
    applied=np.clip(proposal,lo,hi)
    assert np.isfinite(pair).all() and np.isfinite(absolute).all()
    cells=[]
    for ids,n,coverage in zip(physical['cells'],REQUESTED_COUNTS,physical['coverage']):
        item=dict(coverage,physical_response_requested_MSE=float(np.sum(pair[ids]**2)/(n*23)),
            absolute_branch_requested_MSE=float(np.sum(absolute[ids]**2)/(n*23)))
        if len(ids):
            item.update(response_valid_MSE=float(np.mean(pair[ids]**2)),
                absolute_valid_MSE=float(np.mean(absolute[ids]**2)),
                preclip_target_RMSE_rad=float(np.sqrt(np.mean((proposal[ids]-physical['target'][ids])**2))),
                applied_target_RMSE_rad=float(np.sqrt(np.mean((applied[ids]-physical['target'][ids])**2))),
                applied_per_joint_RMSE_rad=np.sqrt(np.mean((applied[ids]-physical['target'][ids])**2,axis=0)).tolist(),
                first24_response_MSE=float(np.mean(pair[ids[:24]]**2)),
                student_clipped_rows=int(np.any(applied[ids]!=proposal[ids],axis=1).sum()))
        else:item.update(response_valid_MSE=None,absolute_valid_MSE=None)
        cells.append(item)
    groups={}
    for name in ('policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped',
                 'successor_replan_boundary','successor_zero_gain'):
        mask=physical[name]
        if mask.ndim==2:mask=np.any(mask,axis=1)
        groups[name]=dict(valid_rows=int(mask.sum()),response_MSE=float(np.mean(pair[mask]**2)) if mask.any() else None,
            absolute_MSE=float(np.mean(absolute[mask]**2)) if mask.any() else None)
    return dict(nine_cell_response_objective=float(np.mean([c['physical_response_requested_MSE'] for c in cells])),
        nine_cell_absolute_branch_diagnostic=float(np.mean([c['absolute_branch_requested_MSE'] for c in cells])),
        cells=cells,groups=groups,requested=3054,valid=len(branch_delta),strict_failed=3054-len(branch_delta),
        training_diagnostics_only=True,checkpoint_selection=False,physics_steps=0)
