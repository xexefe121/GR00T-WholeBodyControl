"""Read-only completed-run accounting; no model, optimizer, ORT or native calls."""
import ctypes
from ctypes import wintypes
import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
EXPECTED_LAUNCH='1245984f798beb2ae2ad2c55fe86983fc051219448e2c98149ab36219fce7135'
EXPECTED_TRAINING='ac61a9cfdcb32f4af210664ec20562e78bf3826f85bac4e7cf5e08f300167045'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def process_state(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        code=ctypes.get_last_error();assert code==87,('Process observation failed',pid,code)
        return dict(pid=pid,absent=True,running=False,error=code)
    try:
        value=wintypes.DWORD();assert kernel.GetExitCodeProcess(handle,ctypes.byref(value))
        return dict(pid=pid,absent=False,running=value.value==259,exit_code=value.value)
    finally:kernel.CloseHandle(handle)

def main():
    assert sys.platform=='win32'
    out=BASE/'completion_verification_v1.json';assert not out.exists()
    launch_path=BASE/'fit_launch_receipt.json';training_path=BASE/'training_frozen_inputs.json'
    assert sha(launch_path)==EXPECTED_LAUNCH and sha(training_path)==EXPECTED_TRAINING
    launch=read(launch_path);training=read(training_path)
    root_start=read(BASE/'fit_process/root_launch.json')
    child=read(BASE/'fit_process/child.json');exit_result=read(BASE/'fit_process/exit.json')
    assert root_start['wrapper_pid']==child['wrapper_pid']==exit_result['wrapper_pid']==16876
    assert child['child_pid']==exit_result['child_pid']==21808
    states=[process_state(pid) for pid in (16876,21808)]
    assert all(not p['running'] for p in states),'Wait for existing processes; never rerun fit'
    assert exit_result['exit_code']==exit_result['raw_python_exit_code']==0
    assert exit_result['exit_known'] is True and exit_result['error'] is None
    assert exit_result['launch_receipt_sha256']==EXPECTED_LAUNCH
    assert exit_result['all_postrun_pins_exact'] is True and exit_result['automatic_resume'] is False
    pre=read(BASE/'fit_process/prerun_pins.json');post=read(BASE/'fit_process/postrun_pins.json')
    assert pre['all_exact'] is True and post['all_exact'] is True
    assert pre['count']==post['count']==len(launch['input_sha256'])
    comparisons={}
    for name,digest in launch['input_sha256'].items():
        actual=sha(Path(name));assert actual==digest,name
        for record in (pre,post):
            assert record['files'][name]['expected']==record['files'][name]['actual']==digest
            assert record['files'][name]['matched'] is True
        comparisons[name]=dict(expected=digest,actual=actual,matched=True)
    assert sha(BASE/'training_request.json')==training['training_request_sha256']
    for name,digest in training['source_sha256'].items():assert sha(BASE/'source_snapshot_v1'/name)==digest
    report=read(BASE/'fit/report.json');status=read(BASE/'fit/attempt_status.json')
    for key in ('completed','optimization_completed','final_export_diagnostics_completed',
                'numerical_gate_passed','export_parity_passed'):assert report[key] is True,key
    assert report['ordinary_final_step']==75000 and report['additional_updates']==5000
    assert report['head_ONNX_calls']==1150 and report['final_head_parity']<=1e-5
    assert report['physical_evaluations']==report['new_expert_queries']==0
    assert report['checkpoint_selection'] is False and report['hardware_authorized'] is False
    assert status['stage']=='COMPLETE' and status['completed_steps']==75000
    for counter in (status,report['attempt_counters']):
        assert counter['completed_steps']==75000 and counter['optimizer_step_counters']==[75000]*6
        assert counter['committed_sample_rows']==counter['committed_loss_rows']==5000
        for key,n in dict(training_head_rows=36315000,head_onnx_calls=1150,
                          diagnostic_torch_rows=293480,analytical_head_evaluations=14).items():
            assert counter[key+'_attempted']==counter[key+'_returned']==n,key
        assert counter['legacy_diagnostics']['head_onnx_calls_attempted']==counter['legacy_diagnostics']['head_onnx_calls_returned']==1126
        assert counter['physical_diagnostics']['head_onnx_calls_attempted']==counter['physical_diagnostics']['head_onnx_calls_returned']==24
    restoration=read(BASE/'fit/restoration70000_parity.json')
    for key in ('full_model_AdamW_RNG_exact','all3057_nominal_predictions_and_loss_exact',
                'old_normalization_span_exact','restored_ONNX_bytes_exact'):assert restoration[key] is True,key
    assert restoration['head_ONNX_calls']==575
    arrays=np.load(BASE/'fit/training_arrays.npz',allow_pickle=False)
    assert np.array_equal(arrays['global_step'],np.arange(70001,75001))
    for key in ('nominal_objective','sampled_chord_objective','physical_objective','training_objective','learning_rate'):
        assert arrays[key].shape==(5000,) and np.isfinite(arrays[key]).all(),key
    assert arrays['cell_losses'].shape==(5000,3,9) and np.isfinite(arrays['cell_losses']).all()
    for name,dtype,upper in (('sampled_axes.npy',np.int8,23),('sampled_center_rows.npy',np.int32,3057)):
        values=np.load(BASE/'fit'/name,mmap_mode='r',allow_pickle=False)
        assert values.shape==(5000,576) and values.dtype==dtype
        assert ((values>=0)&(values<upper)).all()
    assert sha(BASE/'fit/student_head.pt')==report['checkpoint_sha256']
    assert sha(BASE/'fit/student_head.onnx')==report['onnx_sha256']
    output_pins={p.relative_to(BASE).as_posix():sha(p) for directory in ('fit','fit_process')
                 for p in sorted((BASE/directory).rglob('*')) if p.is_file()}
    for name in ('final_predictions.npz','initial_predictions.npz'):
        with np.load(BASE/'fit'/name,allow_pickle=False) as z:
            assert z['predicted_delta'].shape==z['onnx_predicted_delta'].shape==(143679,23)
    for name in ('final_physical_predictions.npz','initial_physical_predictions.npz'):
        with np.load(BASE/'fit'/name,allow_pickle=False) as z:
            assert z['predicted_delta'].shape==z['onnx_predicted_delta'].shape==(3054,23)
    # Preserve both initial and final digests: the completed evidence must remain immutable.
    for name,digest in launch['input_sha256'].items():assert sha(Path(name))==digest,name
    for name,digest in output_pins.items():assert sha(BASE/name)==digest,name
    assert sha(launch_path)==EXPECTED_LAUNCH
    result=dict(passed=True,kind='owner_completed_fit_accounting_without_model_evaluation',
        completed_steps=75000,additional_updates=5000,process_states=states,
        training_receipt_sha256=EXPECTED_TRAINING,launch_receipt_sha256=EXPECTED_LAUNCH,
        source_sha256=sha(Path(__file__)),input_count=len(comparisons),input_checks=comparisons,
        output_sha256=output_pins,reported_optimizer_counters=[75000]*6,
        head_ONNX_calls=1150,training_head_rows=36315000,diagnostic_torch_rows=293480,
        final_export_parity=report['final_head_parity'],checkpoint_sha256=report['checkpoint_sha256'],
        onnx_sha256=report['onnx_sha256'],independent_saved_fit_audit_pending=True,
        this_verification_model_evaluations=0,this_verification_optimizer_updates=0,
        this_verification_graph_calls=0,this_verification_native_steps=0,behavioral_qualification=False)
    write(out,result)
    print(json.dumps({k:result[k] for k in ('passed','completed_steps','head_ONNX_calls','input_count',
        'final_export_parity','checkpoint_sha256','onnx_sha256')}))
    print('completion_sha256',sha(out))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        p=BASE/'completion_verification_failure_v1.json'
        if not p.exists():write(p,dict(passed=False,exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
