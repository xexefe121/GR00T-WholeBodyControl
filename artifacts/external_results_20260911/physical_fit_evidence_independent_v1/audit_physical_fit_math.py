"""Independent saved-array math. No trainer imports, models, ORT or physics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def local(value):
    value=str(value).replace('\\','/')
    if value.startswith('/mnt/'):value=value[5].upper()+':'+value[6:]
    return Path(value)

class Proof:
    def __init__(self,destination):
        self.destination=Path(destination);self.count=0;self.stage='preflight';self.context={}
        self.stream=(self.destination/'comparisons.jsonl').open('x',encoding='utf-8')
    def check(self,name,condition):
        self.count+=1
        self.stream.write(json.dumps(dict(number=self.count,name=name,passed=bool(condition),stage=self.stage))+'\n')
        if not condition:
            self.stream.flush()
            if self.context:np.savez_compressed(self.destination/'first_failed_context.npz',**self.context)
            raise AssertionError(name)
    def exact(self,a,b,name):
        a,b=np.asarray(a),np.asarray(b)
        passed=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
        if not passed:np.savez_compressed(self.destination/'first_mismatch.npz',actual=a,expected=b)
        self.check(name,passed)
    def tree(self,a,b,name):
        if isinstance(a,dict):
            self.check(name+' keys',isinstance(b,dict) and set(a)==set(b))
            for k in a:self.tree(a[k],b[k],name+'.'+str(k))
        elif isinstance(a,list):
            self.check(name+' list',isinstance(b,list) and len(a)==len(b))
            for i,(x,y) in enumerate(zip(a,b)):self.tree(x,y,name+f'[{i}]')
        else:self.check(name,type(a)==type(b) and a==b)
    def finish(self):self.stream.flush();self.stream.close()

def identities():
    rows=[(d,c,d*1019+c-250) for d in range(3) for c in range(250,1268)]
    return np.asarray(rows,dtype=np.int64)

def phase_cells(dataset,clock,physical=False):
    start=251 if physical else 250
    return [np.flatnonzero((dataset==d)&(clock>=lo)&(clock<hi)) for d in range(3)
            for lo,hi in ((start,350),(350,1169),(1169,1269))]

def budgets(valid):
    return dict(valid_rows=valid,updates=5000,training_head_rows=5000*(4209+valid),
        legacy_head_onnx_calls=1126,physical_head_onnx_calls=2*((valid+255)//256),
        head_onnx_calls=1126+2*((valid+255)//256),diagnostic_torch_rows=2*(143686+valid),
        analytical_head_evaluations=14,BFM_calls=0,native_steps=0)

def logged_cell_reductions(cell_row):
    """Original scalar Torch reductions: nominalfloat32; other termsfloat64."""
    assert cell_row.shape==(3,9) and cell_row.dtype==np.float64
    return np.asarray([float(torch.from_numpy(cell_row[i].astype(dtype)).mean())
        for i,dtype in enumerate((np.float32,np.float64,np.float64))],np.float64)

def compose_three(nominal,velocity,physical):
    """Nominal has already been exactly promoted to the stored float64 value."""
    return (nominal+velocity)+physical

def reconstruct_physical(arrays,centers,proof):
    ids=identities();d,c,i=ids.T
    for key,value in dict(dataset=d,start_control=c,successor_control=c+1,source_frame=c+11,
        successor_frame=c+12,center_index=i,successor_center_index=i+1).items():proof.exact(arrays[key],value,key)
    for key,center_key in (('nominal_features','features'),('nominal_base_target','base_target'),('nominal_target','expert_target')):
        proof.exact(arrays[key],centers[center_key][i],key+' bound centers')
    valid=arrays['label_valid']
    proof.check('valid bool3054',valid.dtype==np.bool_ and valid.shape==(3054,))
    proof.exact(valid,arrays['policy_status']==1,'valid equals complete physical status')
    proof.check('all requested rows completed or strict failed',np.all(np.isin(arrays['policy_status'],[1,2])))
    proof.check('all nominal verified',arrays['nominal_verified'].dtype==np.bool_ and arrays['nominal_verified'].all())
    proof.check('all nominal statuses complete',np.all(arrays['nominal_status']==1))
    for name in ('valid_steps','attempted_steps','returned_steps'):
        proof.check('all nominal ten '+name,np.all(arrays['nominal_'+name]==10))
        proof.check('valid physical ten '+name,np.all(arrays['policy_'+name][valid]==10))
        proof.check('strict failure recorded step '+name,np.all((arrays['policy_'+name][~valid]>=1)&(arrays['policy_'+name][~valid]<=10)))
    chosen=np.flatnonzero(valid);succ=i[chosen]+1
    state=dict(rows=chosen,valid_mask=valid.copy(),dataset=d[chosen],successor_control=c[chosen]+1,successor=succ)
    for name,key,tail,dtype in (('features','endpoint_features',(1069,),np.float32),
        ('base','endpoint_base_target',(23,),np.float64),('target','label_fixed_map_target',(23,),np.float64),
        ('residual','label_residual_rad',(23,),np.float64),('collector_head_delta','endpoint_head_delta',(23,),np.float32)):
        a=arrays[key]
        proof.check(key+' schema',a.shape==(3054,*tail) and a.dtype==dtype)
        proof.check(key+' valid finite invalid NaN',np.isfinite(a[valid]).all() and np.isnan(a[~valid]).all())
        state[name]=np.asarray(a[chosen]).copy()
    proof.exact(state['target']-state['base'],state['residual'],'fixed label minus new BFM base')
    lo,hi=centers['joint_limits'].T
    proof.check('native bounded fixed labels',np.all((state['target']>=lo)&(state['target']<=hi)))
    proof.exact(arrays['policy_native_target_clipped'],arrays['policy_raw_target']!=arrays['policy_applied_target'],'policy clip masks')
    proof.exact(arrays['teacher_feedback_clipped'][chosen],arrays['teacher_feedback_raw'][chosen]!=arrays['teacher_feedback_correction'][chosen],'teacher feedback masks')
    proof.exact(arrays['teacher_native_clipped'][chosen],arrays['teacher_preclip_target'][chosen]!=arrays['label_fixed_map_target'][chosen],'teacher native masks')
    proof.exact(arrays['successor_replan_boundary'],centers['plan_control'][i+1]!=centers['plan_control'][i],'all replan-boundary flags')
    proof.exact(arrays['successor_zero_gain'],~np.any(centers['gain'][i+1],axis=(1,2)),'all zero-gain flags')
    for key in ('policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped','successor_replan_boundary','successor_zero_gain'):
        state[key]=np.asarray(arrays[key][chosen]).copy()
    state['cells']=phase_cells(state['dataset'],state['successor_control'],True)
    requested=phase_cells(d,c+1,True)
    proof.check('nine requested99/819/100',list(map(len,requested))==[99,819,100]*3)
    state['coverage']=[dict(dataset=j//3,phase=('acquisition','source','return')[j%3],requested=len(requested[j]),
        valid=len(v),strict_failed=len(requested[j])-len(v),effective_mass=len(v)/(9*len(requested[j])),empty=len(v)==0)
        for j,v in enumerate(state['cells'])]
    return state

def nominal_metrics(predictions,centers,cells,span,pairs,step):
    normalized=torch.from_numpy(predictions['normalized_centers'])
    label=torch.from_numpy((centers['residual_rad']/span).astype(np.float32))
    error_norm=(normalized-label)**2
    objective=float(torch.stack([error_norm[ix].mean() for ix in cells]).mean())
    delta=predictions['predicted_delta'][:3057]
    proposal=centers['base_target']+delta
    applied=np.clip(proposal,*centers['joint_limits'].T)
    error=applied-centers['expert_target'];residual_error=delta-centers['residual_rad']
    def subset(ix):
        return dict(rows=len(ix),applied_target_rmse_rad=float(np.sqrt(np.mean(error[ix]**2))),
            applied_target_by_joint_rmse_rad=np.sqrt(np.mean(error[ix]**2,axis=0)).tolist(),
            unclipped_target_rmse_rad=float(np.sqrt(np.mean(residual_error[ix]**2))),
            clipped_rows=int(np.any(proposal[ix]!=applied[ix],axis=-1).sum()))
    groups=[]
    for d in range(3):
        group=cells[d*3:d*3+3]
        groups.append(dict(dataset=d,all=subset(np.concatenate(group)),phases=[subset(ix) for ix in group],
            first24_each_phase=[subset(ix[:24]) for ix in group]))
    paired=[]
    for p in pairs:
        a,b=p['a_global_row'],p['b_global_row']
        assert p['a_control']==int(centers['control'][a]) and p['b_control']==int(centers['control'][b])
        paired.append(dict(p,predicted_target_difference_rad=(applied[b]-applied[a]).tolist(),
            actual_target_difference_rad=(centers['expert_target'][b]-centers['expert_target'][a]).tolist(),
            a_target_error_rad=error[a].tolist(),b_target_error_rad=error[b].tolist(),
            predicted_residual_difference_rad=(delta[b]-delta[a]).tolist(),
            actual_residual_difference_rad=(centers['residual_rad'][b]-centers['residual_rad'][a]).tolist()))
    return dict(global_step=step,nine_cell_nominal_objective=objective,all=subset(np.arange(3057)),datasets=groups,
        query250_signed_error_rad=error[2038].tolist(),query250_target_rmse_rad=float(np.sqrt(np.mean(error[2038]**2))),
        fixed_pairs=paired,predictions_reused_without_additional_inference=True)

def velocity_metrics(delta,centers,base,teacher,cells,span):
    nominal=centers['base_target']+delta[:3057]
    perturbed=base+delta[3057:].reshape(3057,23,2,23)
    expected=teacher-centers['expert_target'][:,None,None]
    response=perturbed-nominal[:,None,None]
    error=response-expected
    normalized=(error/span)**2
    applied=np.clip(perturbed,*centers['joint_limits'].T)
    nominal_applied=np.clip(nominal,*centers['joint_limits'].T)
    applied_error=applied-nominal_applied[:,None,None]-expected
    rows=[]
    def core(ix):
        return dict(rows=len(ix),axis_pairs=len(ix)*23,
            normalized_proposal_chord_MSE=float(np.mean(normalized[ix])),
            proposal_chord_error_rms_rad=float(np.sqrt(np.mean(error[ix]**2))),
            applied_chord_error_rms_rad=float(np.sqrt(np.mean(applied_error[ix]**2))),
            student_clipped_probes=int(np.any(perturbed[ix]!=applied[ix],axis=-1).sum()))
    for j,ix in enumerate(cells):
        rows.append(dict(dataset=j//3,phase=j%3,**core(ix),
            student_clipped_components=int(np.sum(perturbed[ix]!=applied[ix])),first24=core(ix[:24])))
    return dict(full_nine_cell_chord_objective=float(np.mean([r['normalized_proposal_chord_MSE'] for r in rows])),
        cells=rows,all_probe_axes_and_signs_evaluated=True,checkpoint_selection=False)

def physical_metrics(center_delta,branch_delta,centers,physical,span):
    """Independent full saved-endpoint response, absolute, group and fixed-window metrics."""
    nominal=centers['base_target'][physical['successor']]+center_delta[physical['successor']]
    proposal=physical['base']+branch_delta
    teacher_difference=physical['target']-centers['expert_target'][physical['successor']]
    response=((proposal-nominal)-teacher_difference)/span
    absolute=(proposal-physical['target'])/span
    applied=np.clip(proposal,*centers['joint_limits'].T)
    cells=[]
    for number,ix in enumerate(physical['cells']):
        n=(99,819,100)[number%3]
        value=dict(physical['coverage'][number],physical_response_requested_MSE=float(np.sum(response[ix]**2)/(n*23)),
            absolute_branch_requested_MSE=float(np.sum(absolute[ix]**2)/(n*23)))
        if len(ix):
            value.update(response_valid_MSE=float(np.mean(response[ix]**2)),absolute_valid_MSE=float(np.mean(absolute[ix]**2)),
                preclip_target_RMSE_rad=float(np.sqrt(np.mean((proposal[ix]-physical['target'][ix])**2))),
                applied_target_RMSE_rad=float(np.sqrt(np.mean((applied[ix]-physical['target'][ix])**2))),
                applied_per_joint_RMSE_rad=np.sqrt(np.mean((applied[ix]-physical['target'][ix])**2,axis=0)).tolist(),
                student_clipped_rows=int(np.any(proposal[ix]!=applied[ix],axis=1).sum()))
        else:value.update(response_valid_MSE=None,absolute_valid_MSE=None)
        start=(251,350,1169)[number%3]
        window=ix[(physical['successor_control'][ix]>=start)&(physical['successor_control'][ix]<=start+23)]
        value['first24_requested_window']=dict(successor_control_range=[start,start+23],requested=24,valid=len(window),
            strict_failed=24-len(window),response_valid_MSE=float(np.mean(response[window]**2)) if len(window) else None)
        cells.append(value)
    groups={}
    for key in ('policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped','successor_replan_boundary','successor_zero_gain'):
        mask=physical[key]
        if mask.ndim>1:mask=np.any(mask,axis=1)
        groups[key]=dict(valid_rows=int(mask.sum()),response_MSE=float(np.mean(response[mask]**2)) if mask.any() else None,
            absolute_MSE=float(np.mean(absolute[mask]**2)) if mask.any() else None)
    return dict(nine_cell_response_objective=float(np.mean([r['physical_response_requested_MSE'] for r in cells])),
        nine_cell_absolute_branch_diagnostic=float(np.mean([r['absolute_branch_requested_MSE'] for r in cells])),
        cells=cells,groups=groups,requested=3054,valid=len(branch_delta),strict_failed=3054-len(branch_delta),
        training_diagnostics_only=True,checkpoint_selection=False,physics_steps=0)
