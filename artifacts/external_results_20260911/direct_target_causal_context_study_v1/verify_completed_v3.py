"""Saved-only pair accounting; numerical completion is a separate verdict."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
from pathlib import Path
BASE=Path(__file__).resolve().parent
CONDITIONS=('blinded','causal')
BACKENDS=('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical_pins(pins):
    result={}
    for path,digest in pins.items():
        key=Path(path).resolve().as_posix().casefold()
        if key in result:raise ValueError('Duplicate physical pin path.')
        result[key]=digest
    return result
def validate_process_pin_report(record,expected_pins):
    if record['all_exact'] is not True or record['count']!=len(expected_pins):raise ValueError('Process pin count/pass differs.')
    actual={path:item['expected'] for path,item in record['files'].items()}
    if canonical_pins(actual)!=canonical_pins(expected_pins):raise ValueError('Process pin membership differs.')
    for path,item in record['files'].items():
        if item['matched'] is not True or item['actual']!=item['expected'] or sha(path)!=item['expected']:raise ValueError('Process pin record differs: '+path)
def condition_subjects(base,request,condition):
    if condition not in CONDITIONS:raise ValueError('Explicit condition required.')
    fit=base/'fit';dest=fit/condition;shared=fit/'shared'
    paths=dict(fit_report=dest/'report.json',checkpoint=dest/'student_head.pt',head=dest/'student_head.onnx',
        normalization=shared/'normalization.npz',training_manifest=base/'training_frozen_inputs.json',training_request=base/'training_request.json',
        export_manifest=dest/'output_manifest.json',coefficient=request['subjects']['coefficient_source']['path'],source_checkpoint=request['subjects']['checkpoint']['path'],
        full_state_generation_request=request['full_state_paths']['request'],full_state_generation_report=request['full_state_paths']['report'],
        full_state_data_audit=request['subjects']['full_state_root_audit']['path'],full_state_data_owner=request['subjects']['full_state_root_owner']['path'],
        paired_report=fit/'paired_report.json',shared_manifest=shared/'output_manifest.json',context_alignment=shared/'context_alignment.json',
        blinded_fit_report=fit/'blinded/report.json',causal_fit_report=fit/'causal/report.json')
    return {key:sha(path) for key,path in paths.items()}
def absent(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD);kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD));kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,);kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:
        if ctypes.get_last_error()==87:return True
        raise OSError(ctypes.get_last_error(),'Cannot prove process absence')
    try:
        code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise OSError(ctypes.get_last_error(),'Cannot inspect process')
        return code.value!=259
    finally:kernel.CloseHandle(handle)
def expected(calls,rows):return dict(calls_attempted=calls,calls_returned=calls,calls_synchronized=calls,calls_verified=calls,rows_attempted=rows,rows_returned=rows,rows_verified=rows)
def validate_counter(value,calls,rows,complete):
    if set(value)!=set(expected(0,0)):raise ValueError('Counter keys differ.')
    if any(type(v)!=int or v<0 for v in value.values()):raise ValueError('Counter must be nonnegative integers.')
    c=[value['calls_'+k] for k in ('attempted','returned','synchronized','verified')]
    r=[value['rows_'+k] for k in ('attempted','returned','verified')]
    if c!=sorted(c,reverse=True) or r!=sorted(r,reverse=True) or c[0]>calls or r[0]>rows:raise ValueError('Counter prefix/budget differs.')
    if complete and value!=expected(calls,rows):raise ValueError('Incomplete counter on completed condition.')
def validate_report(report,condition):
    complete=report.get('completed') is True
    counts=report['counts']
    validate_counter(counts['training'],9000,44058000,complete)
    for backend in BACKENDS:validate_counter(counts['diagnostics'][backend],1437,367570,complete)
    for name in ('calibration_forward_calls','calibration_gradient_calls','BFM_calls','native_calls','manual_export_trace_calls'):
        if counts[name]!=0:raise ValueError('Unexpected calls: '+name)
    if complete:
        for name in ('optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged','fresh_optimizer'):
            if report[name] is not True:raise ValueError('Inconsistent completed condition: '+name)
        values=dict(condition=condition,ordinary_final_step=68000,additional_updates=3000,optimizer_step=3000,features=1323,context_features=323,
            head_output='normalized_target',fixed_full_state_coefficient=1.8188207859141674,execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',
            parity_tolerance_rad=1e-5,checkpoint_selection=False,native_steps=0,BFM_calls=0)
        for name,value in values.items():
            if report.get(name)!=value:raise ValueError('Condition contract differs: '+name)
        error=report['max_preclip_error_rad']
        if not math.isfinite(error) or error<0 or error>1e-5:raise ValueError('Numerical parity failed.')
    return complete
def validate_manifest(root,path):
    value=read(path)
    for name,digest in value['files'].items():
        p=(root/name).resolve()
        if p.parent!=root.resolve() or sha(p)!=digest:raise ValueError('Output manifest mismatch: '+str(p))
    return value
def main():
    fit=BASE/'fit';process=BASE/'fit_process_v2'
    result=dict(owner_verification_passed=False,accounting_passed=False,paired_completion_passed=False,numerical_completion_passed=False,
        task_model_calls=0,optimizer_updates=0,native_steps=0,behavioral_qualification=False)
    try:
        exit_report=read(process/'exit.json');start=read(process/'start.json');child=read(process/'child.json')
        pids=[exit_report['wrapper_pid'],exit_report['child_pid']]
        if pids!=[start['wrapper_pid'],child['child_pid']] or child['wrapper_pid']!=pids[0] or child['captured_handle_nonzero'] is not True:raise ValueError('Actual process identities differ.')
        if any(type(p)!=int or p<=0 for p in pids) or not all(absent(p) for p in pids):raise ValueError('Processes still present or unknown.')
        known=exit_report['exit_known'] is True and type(exit_report['raw_python_exit_code'])==int
        if not known or exit_report['child_started'] is not True:raise ValueError('Actual child exit unknown.')
        receipt=read(BASE/'training_frozen_inputs.json');request=read(BASE/'training_request.json');clear=read(BASE/'training_clearance.json')
        pins=dict(receipt['input_sha256']);pins.update({(Path(receipt['source_directory'])/k).as_posix():v for k,v in receipt['source_sha256'].items()})
        for p,digest in pins.items():
            if sha(p)!=digest:raise ValueError('Current frozen pin differs: '+p)
        for name,key in [('training_request.json','request_sha256'),('training_frozen_inputs.json','frozen_receipt_sha256')]:
            if sha(BASE/name)!=clear[key] or start[key]!=clear[key]:raise ValueError('Clearance/request identity differs.')
        if sha(BASE/'training_clearance.json')!=start['clearance_sha256'] or exit_report['clearance_sha256']!=start['clearance_sha256']:raise ValueError('Clearance changed.')
        if sha(clear['review_path'])!=clear['review_sha256'] or sha(clear['launcher_path'])!=clear['launcher_sha256']:raise ValueError('Concrete release changed.')
        launch_pins=dict(pins)
        for name,digest in [('training_request.json',clear['request_sha256']),('training_frozen_inputs.json',clear['frozen_receipt_sha256']),('training_clearance.json',start['clearance_sha256'])]:launch_pins[(BASE/name).as_posix()]=digest
        launch_pins[clear['review_path']]=clear['review_sha256']
        for name in ('prerun_pins.json','postrun_pins.json'):validate_process_pin_report(read(process/name),launch_pins)
        direct=dict(training_request=sha(BASE/'training_request.json'),training_frozen_inputs=sha(BASE/'training_frozen_inputs.json'),
            frozen=sha(BASE/'training_frozen_inputs.json'),process_exit=sha(process/'exit.json'),source_checkpoint=request['subjects']['checkpoint']['sha256'])
        reports={};completions={}
        for condition in CONDITIONS:
            dest=fit/condition
            if not dest.exists():continue
            report=read(dest/'report.json');reports[condition]=report;completions[condition]=validate_report(report,condition)
            direct[condition+'_report']=sha(dest/'report.json')
            for name,key in [('student_head.pt','checkpoint'),('student_head.onnx','head'),('output_manifest.json','output_manifest')]:
                if (dest/name).exists():direct[condition+'_'+key]=sha(dest/name)
            if completions[condition]:
                if report['training_request_sha256']!=direct['training_request'] or report['frozen_receipt_sha256']!=direct['frozen']:raise ValueError('Condition input bindings differ.')
                if direct[condition+'_checkpoint']!=report['checkpoint_sha256'] or direct[condition+'_head']!=report['onnx_sha256'] or direct[condition+'_output_manifest']!=report['output_manifest_sha256']:raise ValueError('Condition outputs differ.')
                validate_manifest(dest,dest/'output_manifest.json')
            elif not (dest/'failure.json').exists():raise ValueError('Incomplete condition missing failure evidence.')
        shared=fit/'shared'
        if (shared/'output_manifest.json').exists():validate_manifest(shared,shared/'output_manifest.json');direct['shared_manifest']=sha(shared/'output_manifest.json')
        if (shared/'normalization.npz').exists():direct['shared_normalization']=sha(shared/'normalization.npz')
        pair_complete=False
        if (fit/'paired_report.json').exists():
            pair=read(fit/'paired_report.json');direct['paired_report']=sha(fit/'paired_report.json')
            if pair['completed'] is not True or pair['conditions']!=list(CONDITIONS) or set(reports)!=set(CONDITIONS) or not all(completions.values()):raise ValueError('False completed-pair claim.')
            for key in ('shared_initial_actor_optimizer_RNG_normalization_exact','shared_schedule_exact','all_frozen_inputs_unchanged'):
                if pair[key] is not True:raise ValueError('Pair shared proof failed.')
            if pair['budgets']!=request['budgets'] or pair['fixed_coefficient']!=request['coefficient'] or pair['condition_report_sha256']!={c:direct[c+'_report'] for c in CONDITIONS}:raise ValueError('Pair contract/report differs.')
            if exit_report['raw_python_exit_code']!=0 or exit_report['exit_code']!=0 or exit_report['error'] is not None:raise ValueError('Completed pair exit failed.')
            for report in reports.values():
                if report['normalization_sha256']!=direct['shared_normalization'] or report['shared_manifest_sha256']!=direct['shared_manifest']:raise ValueError('Shared output differs.')
            pair_complete=True
        else:
            failure=read(fit/'paired_failure.json');direct['paired_failure']=sha(fit/'paired_failure.json')
            if failure['completed'] is not False or failure['automatic_retry'] is not False or exit_report['raw_python_exit_code']==0 or exit_report['exit_code']==0:raise ValueError('Untruthful failed-pair verdict.')
        result.update(owner_verification_passed=True,accounting_passed=True,paired_completion_passed=pair_complete,
            numerical_completion_passed=pair_complete,direct_subject_sha256=direct,condition_completion=completions,
            expected_pids=pids,processes_absent=True,raw_exit_known=known,raw_python_exit_code=exit_report['raw_python_exit_code'],
            exit_code=exit_report['exit_code'],all_postrun_pins_exact=True,current_pin_count=len(pins))
    except BaseException as exc:result['error']=repr(exc)
    owner=BASE/'owner_completion_verification_v3.json'
    with owner.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    if result['paired_completion_passed']:
        for condition in CONDITIONS:
            role_report=dict(result,condition=condition,direct_subject_sha256=condition_subjects(BASE,request,condition),
                pair_owner_sha256=sha(owner),condition_qualification_only=True,controller_selected=False)
            with (BASE/(condition+'_owner_completion_verification.json')).open('x',encoding='utf-8') as f:json.dump(role_report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(accounting_passed=result['accounting_passed'],paired_completion_passed=result['paired_completion_passed'],sha256=sha(owner))))
    if not result['accounting_passed']:raise SystemExit(1)
if __name__=='__main__':main()
