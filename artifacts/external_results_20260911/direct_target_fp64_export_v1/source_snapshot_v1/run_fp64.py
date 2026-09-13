"""One fixed same-weight FP64 numerical validation. No optimizer or native calls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise RuntimeError('Wrong deterministic CUBLAS setting')
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from direct_data import read,sha,load_data
from direct_diagnostics import summarize,parity
from promoted_model import from_checkpoint,export_onnx

BASE=Path(__file__).resolve().parent.parent
CORPORA=('nominal','velocity','physical');SIZES=(9904,140622,3054);BACKENDS=('CPU64','GPU64','ORT64')
SOURCE_KEYS=('features','velocity_features','physical_features');BATCH=256
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    for attempt in range(41):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt==40:raise
            time.sleep(.05)
def check_frozen(receipt):
    for path,digest in receipt['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed frozen input '+path)
    for name,digest in receipt['source_sha256'].items():
        if sha(Path(receipt['source_directory'])/name)!=digest:raise ValueError('Changed frozen source '+name)
def check_consumed_roles(request,receipt):
    pins=receipt['input_sha256']
    for role,subject in request['subjects'].items():
        path=Path(subject['path']).as_posix();digest=subject['sha256']
        if pins.get(path)!=digest or sha(path)!=digest:raise ValueError('Consumed subject not frozen exactly: '+role)
    for backend,corpora in request['old_outputs'].items():
        for corpus,source_path in corpora.items():
            path=Path(source_path).as_posix()
            if path not in pins or sha(path)!=pins[path]:raise ValueError('Consumed old output not frozen exactly: '+backend+'/'+corpus)
def gate():
    request=read(BASE/'export_request.json');receipt=read(BASE/'export_frozen_inputs.json');clearance=read(BASE/'export_clearance.json')
    identity={k:sha(BASE/name) for k,name in [('request','export_request.json'),('receipt','export_frozen_inputs.json'),('clearance','export_clearance.json')]}
    if Path(receipt['source_directory']).resolve()!=Path(__file__).resolve().parent:raise ValueError('Wrong frozen source directory')
    if receipt['export_request_sha256']!=identity['request'] or clearance['request_sha256']!=identity['request'] or clearance['frozen_receipt_sha256']!=identity['receipt'] or clearance['approved'] is not True:raise ValueError('Uncleared request/receipt')
    if sha(clearance['review_path'])!=clearance['review_sha256']:raise ValueError('Changed review')
    review=read(clearance['review_path'])
    if review['prelaunch_review_pass'] is not True or review['export_request_sha256']!=identity['request'] or review['frozen_receipt_sha256']!=identity['receipt']:raise ValueError('Review does not bind exact export')
    if sha(clearance['launcher_path'])!=clearance['launcher_sha256']:raise ValueError('Changed launcher')
    if request['kind']!='one_same_weight_fp64_export_validation' or request['root_selected'] is not True or request['ordinary_final_step']!=55000 or request['parity_tolerance_rad']!=1e-5:raise ValueError('Wrong selected scope')
    if request['backend_order']!=list(BACKENDS) or request['budgets']!=dict(Torch_calls=1202,Torch_rows=307160,ORT_calls=601,ORT_rows=153580,trace_forward_calls=0,optimizer_updates=0,BFM_calls=0,native_steps=0):raise ValueError('Unexpected fixed budget')
    if Path(sys.executable).resolve()!=Path(request['runtime']['python_path']).resolve():raise ValueError('Wrong isolated runtime')
    check_frozen(receipt);check_consumed_roles(request,receipt)
    import onnx,onnxruntime as ort
    if np.__version__!=request['runtime']['numpy_version'] or ort.__version__!=request['runtime']['onnxruntime_version'] or onnx.__version__!=request['runtime']['onnx_version']:raise ValueError('Fixed NumPy/ORT/ONNX version differs')
    return request,receipt,identity
def configure(request):
    r=request['runtime'];torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if torch.__version__!=r['torch_version'] or torch.version.cuda!=r['cuda_version'] or not torch.cuda.is_available():raise ValueError('Selected CUDA runtime unavailable')
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,device=torch.cuda.get_device_name(0),
        deterministic_algorithms=True,TF32=False,AMP=False,CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'])
def raw_targets(value,data):return data['default']+data['span'].astype(np.float64)*np.asarray(value).astype(np.float64)
def run_backend(backend,model,session,data,dest,counters,active,ledger,outputs):
    for corpus,size,key in zip(CORPORA,SIZES,SOURCE_KEYS):
        inputs=data[key]
        if inputs.shape!=(size,1000) or inputs.dtype!=np.float32:raise ValueError('Original corpus features changed')
        out=np.lib.format.open_memmap(dest/(backend+'_'+corpus+'.npy'),mode='w+',dtype=np.float32,shape=(size,23));out[:]=np.nan
        outputs[backend][corpus]=out
        try:
            for start in range(0,size,BATCH):
                stop=min(start+BATCH,size);n=stop-start;active.clear();active.update(backend=backend,corpus=corpus,start=start,stop=stop,input=inputs[start:stop].copy(),returned=False,synchronized=False,verified=False)
                c=counters[backend]
                if c['calls_attempted']>=601:raise ValueError('Fixed per-backend call budget exhausted')
                c['calls_attempted']+=1;c['rows_attempted']+=n
                try:
                    if backend=='ORT64':
                        returned=session.run(None,{'features':inputs[start:stop]});c['calls_returned']+=1;c['rows_returned']+=n;active['returned']=True
                        active['all_outputs']=[np.asarray(v).copy() for v in returned]
                        c['calls_synchronized']+=1;active['synchronized']=True
                        if len(returned)!=1:raise ValueError('ORT output count')
                        actual=np.asarray(returned[0]).copy()
                    else:
                        tensor=torch.from_numpy(inputs[start:stop]).to('cuda' if backend=='GPU64' else 'cpu')
                        with torch.inference_mode():returned=model(tensor)
                        c['calls_returned']+=1;c['rows_returned']+=n;active['returned']=True;active['returned_tensor']=returned
                        if backend=='GPU64':torch.cuda.synchronize()
                        c['calls_synchronized']+=1;active['synchronized']=True;actual=returned.detach().cpu().numpy().copy()
                    active['output']=actual
                    if actual.shape!=(n,23) or actual.dtype!=np.float32:raise ValueError('Public output schema changed')
                    out[start:stop]=actual
                    if not np.isfinite(actual).all():raise ValueError('Nonfinite output')
                    c['calls_verified']+=1;c['rows_verified']+=n;active['verified']=True
                finally:
                    ledger.write(json.dumps({k:active[k] for k in ('backend','corpus','start','stop','returned','synchronized','verified')})+'\n');ledger.flush()
                if c['calls_attempted']%25==0:out.flush();atomic(dest/'progress.json',dict(stage=backend,corpus=corpus,stop=stop,counters=counters))
        finally:out.flush();ledger.flush();os.fsync(ledger.fileno())
def save_active(active,path):
    arrays={}
    for key,value in active.items():
        if torch.is_tensor(value):arrays[key]=value.detach().cpu().numpy().copy()
        elif isinstance(value,list):
            for i,v in enumerate(value):arrays[key+'_'+str(i)]=np.asarray(v).copy()
        else:arrays[key]=np.asarray(value)
    np.savez_compressed(path,**arrays)
def manifest(dest,complete):
    files={}
    for p in sorted(dest.iterdir()):
        if not p.is_file() or p.name in ('manifest.json','report.json','progress.json') or p.suffix=='.tmp':continue
        item=dict(path=p.name,sha256=sha(p))
        if p.suffix=='.npy':
            a=np.load(p,mmap_mode='r',allow_pickle=False);item.update(shape=list(a.shape),dtype=str(a.dtype))
        files[p.name]=item
    return dict(completed=complete,files=files)
def main():
    request,receipt,identity=gate();dest=BASE/'export';dest.mkdir(exist_ok=False)
    counters={b:dict(calls_attempted=0,calls_returned=0,calls_synchronized=0,calls_verified=0,rows_attempted=0,rows_returned=0,rows_verified=0) for b in BACKENDS}
    active={};outputs={b:{} for b in BACKENDS};stage='load';started=time.perf_counter();ledger=None
    try:
        data=load_data(request['paths'],receipt['input_sha256']);runtime=configure(request);write(dest/'runtime.json',runtime)
        source=request['subjects'];oldreport=read(source['fit_report']['path'])
        if oldreport['optimization_completed'] is not True or oldreport['completed'] is not False or oldreport['numerical_gate_passed'] is not False:raise ValueError('Original failed release status changed')
        training_audit=read(source['training_audit']['path'])
        if training_audit['evidence_audit_passed'] is not True or training_audit['optimization_evidence_verified'] is not True or training_audit['export_qualified'] is not False:raise ValueError('Missing positive training-evidence / failed original release audit')
        saved=torch.load(source['checkpoint']['path'],map_location='cpu',weights_only=True);model=from_checkpoint(saved);model.eval()
        if list(model.parameters()):raise ValueError('Promoted validation must have no trainable parameters')
        promoted={k:v.cpu().numpy().copy() for k,v in model.state_dict().items()}
        for label,value in [('feature_mean',saved['feature_mean']),('feature_std',saved['feature_std'])]:
            if not np.array_equal(promoted[label],value.cpu().numpy().astype(np.float64)):raise ValueError('Promoted normalization differs')
        for i,index in enumerate((0,2,4)):
            for name,key in [('w','weight'),('b','bias')]:
                original=saved['actor_state'][str(index)+'.'+key].cpu().numpy()
                if not np.array_equal(promoted[name+str(i)],original.astype(np.float64)) or promoted[name+str(i)].astype(np.float32).tobytes()!=original.tobytes():raise ValueError('Promoted parameter differs')
        with np.load(source['normalization']['path'],allow_pickle=False) as norm:
            for key in ('feature_mean','feature_std'):
                if norm[key].tobytes()!=saved[key].cpu().numpy().tobytes():raise ValueError('Original norm archive differs')
        for key,value in [('joint_span',data['span']),('default_q',data['default']),('joint_limits',data['limits'])]:
            if saved[key].cpu().numpy().tobytes()!=value.tobytes():raise ValueError('Original physical output contract differs')
        np.savez_compressed(dest/'promoted_parameters.npz',**promoted)
        write(dest/'promotion.json',dict(passed=True,checkpoint_sha256=source['checkpoint']['sha256'],all_source_f32_values_exactly_promoted=True,trainable_parameters=0,optimizer_updates=0))
        stage='manual_export';export_onnx(model,dest/'student_head_fp64.onnx')
        ledger=(dest/'graph_calls.jsonl').open('x',encoding='utf-8')
        stage='CPU64';run_backend(stage,model,None,data,dest,counters,active,ledger,outputs)
        model=model.to('cuda');stage='GPU64';run_backend(stage,model,None,data,dest,counters,active,ledger,outputs)
        import onnxruntime as ort
        if ort.__version__!=request['runtime']['onnxruntime_version']:raise ValueError('ORT version differs')
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        session=ort.InferenceSession(str(dest/'student_head_fp64.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        stage='ORT64';run_backend(stage,None,session,data,dest,counters,active,ledger,outputs);ledger.close();ledger=None
        for c in counters.values():
            if any(c[k]!=601 for k in ('calls_attempted','calls_returned','calls_synchronized','calls_verified')) or any(c[k]!=153580 for k in ('rows_attempted','rows_returned','rows_verified')):raise ValueError('Fixed per-backend budget mismatch')
        stage='saved_output_diagnostics';numerical=parity(outputs['GPU64'],outputs['CPU64'],outputs['ORT64'],data,tolerance=1e-5);write(dest/'parity.json',numerical)
        for backend in BACKENDS:write(dest/('metrics_'+backend+'.json'),summarize(outputs[backend],data))
        drift={}
        for new_backend in BACKENDS:
            for old_backend in ('GPU','CPU','ORT'):
                for corpus in CORPORA:
                    old=np.load(request['old_outputs'][old_backend][corpus],mmap_mode='r',allow_pickle=False)
                    current=np.asarray(outputs[new_backend][corpus]);new_raw=raw_targets(current,data);old_raw=raw_targets(old,data);difference=new_raw-old_raw
                    filename='drift_'+new_backend+'_vs_'+old_backend+'32_'+corpus+'.npy';np.save(dest/filename,difference)
                    if not np.isfinite(difference).all():raise ValueError('Nonfinite saved drift')
                    old_mask=old_raw!=np.clip(old_raw,data['limits'][:,0],data['limits'][:,1]);new_mask=new_raw!=np.clip(new_raw,data['limits'][:,0],data['limits'][:,1]);changed=old_mask!=new_mask
                    drift[filename]=dict(max_abs_preclamp_rad=float(np.max(np.abs(difference))),RMS_preclamp_rad=float(np.sqrt(np.mean(difference**2))),shape=list(difference.shape),
                        old_clipped_rows=int(old_mask.any(axis=1).sum()),old_clipped_components=int(old_mask.sum()),new_clipped_rows=int(new_mask.any(axis=1).sum()),new_clipped_components=int(new_mask.sum()),
                        clipping_changed_rows=int(changed.any(axis=1).sum()),clipping_changed_components=int(changed.sum()))
        write(dest/'drift.json',drift)
        stage='final_rehash'
        for key,name in [('request','export_request.json'),('receipt','export_frozen_inputs.json'),('clearance','export_clearance.json')]:
            if sha(BASE/name)!=identity[key]:raise ValueError('Changed final identity '+key)
        check_frozen(receipt)
        write(dest/'manifest.json',manifest(dest,True))
        passed=bool(numerical['passed']);report=dict(completed=passed,validation_completed=True,numerical_gate_passed=passed,export_parity_passed=passed,optimization_completed_source=True,
            ordinary_final_step=55000,features=1000,head_output='normalized_target',execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',
            ELU_implementation='Where(x>0,x,Exp(Min(x,0))-1)',parity_tolerance_rad=1e-5,max_preclip_error_rad=numerical['maximum_preclamp_rad'],
            checkpoint_sha256=source['checkpoint']['sha256'],source_fit_report_sha256=source['fit_report']['sha256'],source_onnx_sha256=source['source_head']['sha256'],normalization_sha256=source['normalization']['sha256'],onnx_sha256=sha(dest/'student_head_fp64.onnx'),
            export_request_sha256=identity['request'],frozen_receipt_sha256=identity['receipt'],manifest_sha256=sha(dest/'manifest.json'),graph_calls_sha256=sha(dest/'graph_calls.jsonl'),counters=counters,
            all_frozen_inputs_unchanged=True,checkpoint_selection=False,optimizer_updates=0,BFM_calls=0,native_steps=0,trace_forward_calls=0,elapsed_seconds=time.perf_counter()-started)
        write(dest/'report.json',report)
        if not passed:raise ValueError('Selected1e-5 FP64 numerical release failed; all evidence preserved')
        stage='COMPLETE';atomic(dest/'progress.json',dict(stage=stage,counters=counters));print(json.dumps(report),flush=True)
    except BaseException as exc:
        errors=[]
        if ledger is not None:
            try:ledger.flush();os.fsync(ledger.fileno());ledger.close()
            except BaseException as e:errors.append('ledger:'+repr(e))
        for backend in outputs.values():
            for a in backend.values():
                try:a.flush()
                except BaseException as e:errors.append('output_flush:'+repr(e))
        try:save_active(active,dest/'failed_active.npz')
        except BaseException as e:errors.append('active_capture:'+repr(e))
        write(dest/'failure.json',dict(error=repr(exc),stage=stage,counters=counters,preservation_errors=errors,automatic_retry=False))
        try:write(dest/'partial_manifest.json',manifest(dest,False))
        except BaseException as e:errors.append('partial_manifest:'+repr(e))
        if not (dest/'report.json').exists():write(dest/'report.json',dict(completed=False,validation_completed=False,numerical_gate_passed=False,export_parity_passed=False,error=repr(exc),stage=stage,counters=counters,automatic_retry=False,optimizer_updates=0,BFM_calls=0,native_steps=0))
        raise
if __name__=='__main__':main()
