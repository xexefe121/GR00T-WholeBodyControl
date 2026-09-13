"""Bounded release gates and preservation helpers; no training entry point."""
from pathlib import Path
import copy
import json
import os
import random
import sys
import numpy as np
import torch
from direct_data import read, sha
from full_state_contract import SEED
from full_state_diagnostics import atomic

def write(path, value):
    with Path(path).open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def frozen_check(receipt):
    for path,digest in receipt['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed frozen input: '+path)
    for name,digest in receipt['source_sha256'].items():
        if sha(Path(receipt['source_directory'])/name)!=digest:raise ValueError('Changed source: '+name)

def gate(base, source_file):
    files={key:base/name for key,name in [('request','training_request.json'),('frozen','training_frozen_inputs.json'),('clearance','training_clearance.json')]}
    identity={key:sha(path) for key,path in files.items()}
    request=read(files['request']);receipt=read(files['frozen']);clearance=read(files['clearance'])
    if Path(receipt['source_directory']).resolve()!=Path(source_file).parent.resolve():raise ValueError('Not the frozen source directory.')
    if receipt['training_request_sha256']!=identity['request']:raise ValueError('Request identity differs.')
    frozen_check(receipt)
    if clearance['approved'] is not True or clearance['request_sha256']!=identity['request'] or clearance['frozen_receipt_sha256']!=identity['frozen']:raise ValueError('Concrete fit clearance absent.')
    if sha(clearance['review_path'])!=clearance['review_sha256'] or sha(clearance['launcher_path'])!=clearance['launcher_sha256']:raise ValueError('Review or launcher changed.')
    review=read(clearance['review_path'])
    if review['prelaunch_review_pass'] is not True or review['training_request_sha256']!=identity['request'] or review['frozen_receipt_sha256']!=identity['frozen']:raise ValueError('Concrete reviewer bindings differ.')
    if request['kind']!='one_full_state_finite_feedback_fit' or request['root_selected'] is not True or request['updates']!=10000 or request['ordinary_final_step']!=65000 or request['sampler_seed']!=SEED:raise ValueError('Unexpected selected fit.')
    expected=dict(calibration_forward_rows=14686,calibration_forward_calls=3,calibration_gradient_calls=3,
        training_forward_rows=146860000,training_forward_calls=30000,diagnostic_Torch_rows=1470280,
        diagnostic_Torch_calls=5748,diagnostic_ORT_rows=367570,diagnostic_ORT_calls=1437,
        native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
    if request['budgets']!=expected:raise ValueError('Fixed budget differs.')
    for name,subject in request['subjects'].items():
        path=Path(subject['path']).as_posix()
        if receipt['input_sha256'].get(path)!=subject['sha256'] or sha(path)!=subject['sha256']:raise ValueError('Consumed subject not frozen: '+name)
    for name in ('full_state_root_audit','full_state_data_review'):
        subject=request['subjects'][name];qualification=read(subject['path'])
        if qualification[subject['pass_field']] is not True:raise ValueError('Full-state qualification failed: '+name)
        for key,digest in subject['required_fields'].items():
            if qualification.get(key)!=digest:raise ValueError('Full-state qualification subject differs: '+name+'/'+key)
    for name,path in request['full_state_paths'].items():
        if receipt['input_sha256'].get(Path(path).as_posix())!=sha(path):raise ValueError('Unfrozen full-state path: '+name)
    for name,path in request['restoration_predictions'].items():
        if receipt['input_sha256'].get(Path(path).as_posix())!=sha(path):raise ValueError('Unfrozen restored prediction: '+name)
    runtime=request['runtime']
    if Path(sys.executable).resolve()!=Path(runtime['python_path']).resolve():raise ValueError('Wrong isolated Python runtime.')
    if receipt['input_sha256'].get(Path(runtime['verification_path']).as_posix())!=runtime['verification_sha256'] or sha(runtime['verification_path'])!=runtime['verification_sha256'] or read(runtime['verification_path'])[runtime['pass_field']] is not True:raise ValueError('Runtime verification absent.')
    return receipt,request,identity

def final_identity(base,identity,receipt):
    for key,name in [('request','training_request.json'),('frozen','training_frozen_inputs.json'),('clearance','training_clearance.json')]:
        if sha(base/name)!=identity[key]:raise ValueError('Final identity changed: '+key)
    frozen_check(receipt)

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
    import onnx
    import onnxruntime as ort
    runtime=request['runtime'];torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for key,actual in [('torch_version',torch.__version__),('cuda_version',torch.version.cuda),('numpy_version',np.__version__),('onnx_version',onnx.__version__),('ort_version',ort.__version__)]:
        if runtime[key]!=actual:raise ValueError('Runtime version differs: '+key)
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no fallback.')
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
    return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        onnx=onnx.__version__,ORT=ort.__version__,device=torch.cuda.get_device_name(0),deterministic_algorithms=True,
        CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],matmul_TF32=False,cudnn_TF32=False,AMP=False,
        parameter_dtype='float32',optimizer_foreach=False,optimizer_fused=False)

def flatten_active(value,prefix,result):
    if torch.is_tensor(value):result[prefix]=value.detach().cpu().numpy().copy()
    elif isinstance(value,np.ndarray):result[prefix]=value.copy()
    elif isinstance(value,dict):
        for key,item in value.items():flatten_active(item,prefix+'_'+str(key),result)
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):flatten_active(item,prefix+'_'+str(index),result)
    elif value is not None:result[prefix]=np.asarray(value)

def counter():
    return dict(calls_attempted=0,calls_returned=0,calls_synchronized=0,calls_verified=0,
                rows_attempted=0,rows_returned=0,rows_verified=0)

def counted_forward(model,features,name,counts,active,maximum_calls,maximum_rows):
    rows=len(features)
    if counts['calls_attempted']>=maximum_calls or counts['rows_attempted']+rows>maximum_rows:raise ValueError('Forward budget exhausted.')
    # Clear only this current graph's prior result; earlier graph returns stay identifiable.
    active.pop(name+'_prediction',None)
    active['current_forward']=dict(name=name,rows=rows,returned=False,synchronized=False,verified=False)
    counts['calls_attempted']+=1;counts['rows_attempted']+=rows
    result=model(features)
    counts['calls_returned']+=1;counts['rows_returned']+=rows
    active[name+'_prediction']=result.detach();active['current_forward']['returned']=True
    torch.cuda.synchronize();counts['calls_synchronized']+=1;active['current_forward']['synchronized']=True
    if result.shape!=(rows,23) or result.dtype!=torch.float32 or not bool(torch.isfinite(result).all()):raise ValueError('Returned forward schema/nonfinite.')
    counts['calls_verified']+=1;counts['rows_verified']+=rows;active['current_forward']['verified']=True
    return result

def save_loss_prefix(dest,losses,nominal_cells,full_cells,physical_cells):
    for name,values,width in [('training_progress',losses,6),('nominal_cell_losses',nominal_cells,15),
                              ('full_state_cell_losses',full_cells,54),('physical_cell_losses',physical_cells,9)]:
        np.save(dest/(name+'.npy'),np.asarray(values,dtype=np.float64).reshape(-1,width))

def manifest(dest):
    return {path.name:sha(path) for path in sorted(dest.iterdir()) if path.is_file() and path.name not in ('report.json','output_manifest.json','progress.json')}
