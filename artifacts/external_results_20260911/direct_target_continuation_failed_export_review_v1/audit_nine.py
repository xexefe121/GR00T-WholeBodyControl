"""Nine fixed saved-output comparisons; no inference/optimizer/native imports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_continuation_v1')
OUT=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def freeze():
    paths=[Path(__file__),BASE/'training_request.json',BASE/'training_frozen_inputs.json']
    paths.extend(BASE/'fit'/name for name in ['report.json','student_head.pt','student_head.onnx','normalization.npz','data_identities.npz','export_parity.json','failure.json','restoration.json'])
    paths.extend(BASE/'fit_process'/name for name in ['exit.json','postrun_pins.json'])
    paths.extend(BASE/'fit'/f'{provider}_{corpus}.npy' for provider in ('final_GPU','final_CPU','final_ORT') for corpus in ('nominal','velocity','physical'))
    assert sha(BASE/'fit/report.json')=='dc8d834e8b7ef193cb6212248e4c35ca2cd7dab3cfb839a826a3e7adf7d220e6'
    assert sha(BASE/'fit/student_head.pt')=='9ceef5099ebd154e08a1c2c3784c4e06021d464e607474548763665b24f8f9e7'
    assert sha(BASE/'fit/student_head.onnx')=='f9b352ce4cedbbdd70c59f09b8a28080f7a696bfff464621719cc93e6c899797'
    request=dict(kind='nine_fixed_saved_export_comparisons',input_sha256={str(p):sha(p) for p in paths},
        tolerance_rad=1e-5,model_calls=0,optimizer_updates=0,native_steps=0,expected_export_gate_pass=False)
    with (OUT/'request.json').open('x',encoding='utf-8') as f:json.dump(request,f,indent=2);f.write('\n')
    print(json.dumps({'request_sha256':sha(OUT/'request.json')}))
def run():
    request=read(OUT/'request.json');request_sha=sha(OUT/'request.json');checked={}
    for path,digest in request['input_sha256'].items():assert sha(path)==digest,path
    report=read(BASE/'fit/report.json');parity=read(BASE/'fit/export_parity.json');failure=read(BASE/'fit/failure.json');exit_receipt=read(BASE/'fit_process/exit.json')
    assert report['optimization_completed'] is True and report['final_export_diagnostics_completed'] is True
    assert report['completed'] is False and report['numerical_gate_passed'] is False and report['export_parity_passed'] is False
    assert report['ordinary_final_step']==55000 and report['additional_updates']==50000
    assert failure['preservation_errors']==[] and failure['ordinary_final_preserved'] is True
    assert failure['state']['optimizer_step_counters']==[55000]*6
    assert report['counters']==failure['counters']==dict(training_head_rows_attempted=705500000,training_head_rows_returned=705500000,
        diagnostic_torch_rows_attempted=460740,diagnostic_torch_rows_returned=460740,ORT_calls_attempted=601,ORT_calls_returned=601)
    assert exit_receipt['exit_known'] is True and exit_receipt['raw_python_exit_code']==exit_receipt['exit_code']==1
    assert exit_receipt['all_postrun_pins_exact'] is True
    post=read(BASE/'fit_process/postrun_pins.json')
    assert post['all_exact'] is True and post['count']==len(post['files'])==202
    assert all(v['matched'] is True and v['actual']==v['expected'] for v in post['files'].values())
    with np.load(BASE/'fit/normalization.npz',allow_pickle=False) as z:
        span=z['joint_span'].astype(np.float64);default=z['default_q'];limits=z['joint_limits']
    with np.load(BASE/'fit/data_identities.npz',allow_pickle=False) as z:ids={k:z[k].copy() for k in z.files}
    def location(corpus,row,joint):
        if corpus=='velocity':
            center=row//46;axis=(row%46)//2;sign=row%2
            return dict(row=int(row),joint=int(joint),dataset=int(ids['dataset'][center]),control=int(ids['control'][center]),
                source_frame=int(ids['source_frame'][center]),center_row=int(center),velocity_axis=int(axis),sign_index=int(sign))
        if corpus=='physical':
            successor=int(ids['physical_successor'][row])
            return dict(row=int(row),joint=int(joint),dataset=int(ids['physical_dataset'][row]),control=int(ids['physical_control'][row]),
                source_frame=int(ids['source_frame'][successor]),successor_nominal_row=successor)
        return dict(row=int(row),joint=int(joint),dataset=int(ids['dataset'][row]),control=int(ids['control'][row]),source_frame=int(ids['source_frame'][row]))
    comparisons={};arrays={};allmax=0.;threshold=1e-5
    for corpus,count in [('nominal',9904),('velocity',140622),('physical',3054)]:
        raw={}
        for provider in ('final_GPU','final_CPU','final_ORT'):
            output=np.load(BASE/'fit'/f'{provider}_{corpus}.npy',mmap_mode='r',allow_pickle=False)
            assert output.shape==(count,23) and output.dtype==np.float32 and np.isfinite(output).all()
            raw[provider]=default+span*output.astype(np.float64)
        for left,right,name in [('final_GPU','final_CPU','GPU_CPU'),('final_GPU','final_ORT','GPU_ORT'),('final_CPU','final_ORT','CPU_ORT')]:
            key=corpus+'_'+name;errors=np.abs(raw[left]-raw[right]);row,joint=np.unravel_index(np.argmax(errors),errors.shape)
            maximum=float(errors[row,joint]);assert maximum==parity['comparisons'][key];allmax=max(allmax,maximum)
            clipped={p:np.minimum(np.maximum(raw[p],limits[:,0]),limits[:,1]) for p in (left,right)}
            worst=location(corpus,row,joint)
            worst.update(left_preclamp_rad=float(raw[left][row,joint]),right_preclamp_rad=float(raw[right][row,joint]),joint_span=float(span[joint]),
                left_outside_native_range=bool(raw[left][row,joint]<limits[joint,0] or raw[left][row,joint]>limits[joint,1]),
                right_outside_native_range=bool(raw[right][row,joint]<limits[joint,0] or raw[right][row,joint]>limits[joint,1]))
            comparisons[key]=dict(maximum_preclamp_rad=maximum,passes_original_tolerance=maximum<=threshold,
                components_exceeding_tolerance=int(np.count_nonzero(errors>threshold)),rows_exceeding_tolerance=int(np.count_nonzero(np.any(errors>threshold,axis=1))),
                p50_component_rad=float(np.quantile(errors,.5)),p95_component_rad=float(np.quantile(errors,.95)),
                maximum_after_original_native_clamp_rad=float(np.max(np.abs(clipped[left]-clipped[right]))),worst=worst)
            arrays[key+'_row_maximum']=errors.max(axis=1)
    assert allmax==parity['maximum_preclamp_rad'] and parity['passed'] is False and parity['nonfinite_comparisons']==[] and allmax>threshold
    assert report['export_parity']==parity
    changes={key:100*(report['final_metrics'][key]/report['initial_metrics'][key]-1) for key in ['nominal_objective','full_velocity_objective','physical_objective']}
    with (OUT/'row_errors.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    for path,digest in request['input_sha256'].items():assert sha(path)==digest,path
    assert sha(OUT/'request.json')==request_sha
    result=dict(saved_comparisons_verified=True,export_review_pass=False,original_export_gate_pass=False,
        request_sha256=request_sha,input_sha256=request['input_sha256'],row_errors_sha256=sha(OUT/'row_errors.npz'),
        ordinary_final_step=55000,additional_updates=50000,optimization_completed=True,selected_diagnostics_completed=True,
        maximum_preclamp_rad=allmax,original_tolerance_rad=threshold,failed_comparison_count=sum(not v['passes_original_tolerance'] for v in comparisons.values()),
        comparisons=comparisons,objective_change_percent=changes,initial_metrics={k:report['initial_metrics'][k] for k in changes},final_metrics={k:report['final_metrics'][k] for k in changes},
        counters=report['counters'],all_finite=True,preservation_errors=[],exit_code=1,all202_postrun_pins_reported_exact=True,
        all_review_inputs_rehashed=True,model_calls=0,optimizer_updates=0,native_steps=0,
        limitations=['This verifies the failed numerical gate; it does not clear this float32 export for a witness or canonical run.',
                    'Post-clamp errors are descriptive only; the original fixed gate uses preclamp error and is unchanged.',
                    'No provider evaluations, fitting, alternate checkpoints, tolerances, or runtime candidates were executed.'])
    with (OUT/'report.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'saved_comparisons_verified':True,'export_pass':False,'report_sha256':sha(OUT/'report.json'),'maximum':allmax,'changes':changes}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);a=p.parse_args();freeze() if a.mode=='freeze' else run()
