"""Compare two ordinary direct heads from saved output arrays only."""
import hashlib,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def change(a,b):return dict(step5000=float(a),step55000=float(b),difference=float(b-a),percent_change=float(100*(b/a-1)) if a else None,worsened=bool(b>a))
def main():
    old=ROOT/'direct_target_student_v1';new=ROOT/'direct_target_continuation_v1';centers_path=ROOT/'velocity_chord_student_v1/generation/centers.npz'
    paths=[Path(__file__),centers_path]
    for b in (old,new):paths += [b/'fit'/name for name in ('report.json','final_metrics.json','normalization.npz','data_identities.npz','final_GPU_nominal.npy')]
    paths += [old/'owner_completion_verification.json',new/'owner_failure_verification.json']
    pins={p.as_posix():sha(p) for p in paths}
    write(BASE/'request.json',dict(kind='saved_ordinary5000_vs55000_comparison',source_sha256=sha(Path(__file__)),input_sha256=pins,
        subject='same fixed nominal acquisition states; no actual55000 rollout inputs',query_dataset=2,query_controls=list(range(250,350)),
        model_calls=0,optimizer_updates=0,native_steps=0,no_selection=True))
    for b in (old,new):
        if read(b/'fit/report.json')['optimization_completed'] is not True:raise ValueError('Incomplete optimization')
    if read(old/'owner_completion_verification.json')['owner_verification_passed'] is not True or read(new/'owner_failure_verification.json')['owner_evidence_verification_passed'] is not True:raise ValueError('Owner evidence verification missing')
    if read(new/'fit/report.json')['numerical_gate_passed'] is not False:raise ValueError('Expected failed numerical gate must remain explicit')
    if sha(old/'fit/normalization.npz')!=sha(new/'fit/normalization.npz'):raise ValueError('Normalization changed')
    a=read(old/'fit/final_metrics.json');b=read(new/'fit/final_metrics.json')
    summary={k:change(a[k],b[k]) for k in ('nominal_objective','full_velocity_objective','physical_objective')}
    cells={}
    for group,keys in [('nominal_cells',('normalized_MSE','preclip_RMSE_rad','applied_RMSE_rad','clipped_rows','clipped_components')),('velocity_cells',('response_MSE','first24_response_MSE','preclip_RMSE_rad','applied_RMSE_rad','clipped_rows','clipped_components')),('physical_cells',('response_MSE','first24_response_MSE','preclip_RMSE_rad','applied_RMSE_rad','clipped_rows','clipped_components'))]:
        items=[]
        for x,y in zip(a[group],b[group]):
            if (x['dataset'],x['phase'])!=(y['dataset'],y['phase']):raise ValueError('Cell identity changed')
            item=dict(dataset=x['dataset'],phase=x['phase'],metrics={k:change(x[k],y[k]) for k in keys})
            if group=='nominal_cells':item['first24']={k:change(x['first24'][k],y['first24'][k]) for k in ('preclip_RMSE_rad','applied_RMSE_rad','clipped_rows','clipped_components')}
            items.append(item)
        cells[group]=items
    with np.load(old/'fit/normalization.npz') as norm:
        default=norm['default_q'];span=norm['joint_span'].astype(np.float64);limits=norm['joint_limits']
    with np.load(old/'fit/data_identities.npz') as ident:
        dataset=ident['dataset'].copy();control=ident['control'].copy()
    with np.load(new/'fit/data_identities.npz') as ident:
        if not np.array_equal(dataset,ident['dataset']) or not np.array_equal(control,ident['control']):raise ValueError('Nominal row identity changed')
    with np.load(centers_path) as c:
        cids=np.flatnonzero((c['dataset']==2)&(c['control']>=250)&(c['control']<350));teacher=c['expert_target'][cids].copy();cc=c['control'][cids].copy()
    ids=np.flatnonzero((dataset==2)&(control>=250)&(control<350))
    if len(ids)!=100 or not np.array_equal(control[ids],np.arange(250,350)) or not np.array_equal(cc,control[ids]):raise ValueError('Query250 acquisition mapping mismatch')
    outputs={};query_metrics={};rows=[]
    for step,base in [('5000',old),('55000',new)]:
        output=np.load(base/'fit/final_GPU_nominal.npy',mmap_mode='r')[ids].copy()
        raw=default+span*output.astype(np.float64);applied=np.clip(raw,limits[:,0],limits[:,1]);error=raw-teacher;ae=applied-teacher;mask=raw!=applied
        outputs.update({step+'_normalized_output':output,step+'_raw_target':raw,step+'_applied_target':applied,step+'_error':error,step+'_applied_error':ae,step+'_clip_mask':mask})
        query_metrics[step]={}
        for label,slice_ in [('first',slice(0,1)),('first24',slice(0,24)),('all100',slice(None))]:
            query_metrics[step][label]=dict(preclip_RMSE_rad=float(np.sqrt(np.mean(error[slice_]**2))),applied_RMSE_rad=float(np.sqrt(np.mean(ae[slice_]**2))),clipped_rows=int(np.any(mask[slice_],axis=1).sum()),clipped_components=int(mask[slice_].sum()))
    for i,c in enumerate(cc):rows.append(dict(control=int(c),preclip_RMSE_rad=change(np.sqrt(np.mean(outputs['5000_error'][i]**2)),np.sqrt(np.mean(outputs['55000_error'][i]**2))),applied_RMSE_rad=change(np.sqrt(np.mean(outputs['5000_applied_error'][i]**2)),np.sqrt(np.mean(outputs['55000_applied_error'][i]**2)))))
    np.savez_compressed(BASE/'query250_acquisition.npz',control=cc,nominal_row=ids,teacher_target=teacher,**outputs)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Input changed '+path)
    write(BASE/'report.json',dict(passed=True,numerical_gate_passed=False,canonical_evaluation_cleared=False,scope='saved identical nominal acquisition inputs, not counterfactual actual55000 states',summary=summary,cells=cells,query250_acquisition=query_metrics,query250_rows=rows,
        query_arrays_sha256=sha(BASE/'query250_acquisition.npz'),request_sha256=sha(BASE/'request.json'),all_inputs_unchanged=True,model_calls=0,optimizer_updates=0,native_steps=0,no_selection=True,
        limitations=['Saved GPU outputs are used; WSL batch1 witness and connected behavior require separate gates.','All worsening velocity/physical cells and clipping counts are retained; aggregate training improvement does not prove closed-loop stability.']))
    print(json.dumps(dict(passed=True,report_sha256=sha(BASE/'report.json'),summary=summary,query250=query_metrics)))
if __name__=='__main__':main()
