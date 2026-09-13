"""One saved-only owner receipt; completion and process accounting stay separate."""
from pathlib import Path
import json
import math
from execution_common import sha,read,canonical_pins,validate_process_pin_report,absent,validate_counter,BACKENDS
BASE=Path(__file__).resolve().parent

def eq(actual,expected,name):
    if type(actual) is not type(expected) or actual!=expected:raise ValueError('Exact metadata differs: '+name)
def require(value,name):
    if value is not True:raise ValueError(name)
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def validate_report(report,request):
    complete=report.get('completed') is True
    if 'counts' not in report:
        if complete or report.get('task_forward_calls')!=0 or report.get('optimizer_updates')!=0:raise ValueError('Missing truthful setup-failure counters')
        return False
    counts=report['counts'];budget=request['budgets']
    validate_counter(counts['training'],budget['training_forward_calls'],budget['training_forward_rows'],complete)
    for backend in BACKENDS:validate_counter(counts['diagnostics'][backend],1441,368588,complete)
    for name in ('calibration_forward_calls','calibration_gradient_calls','BFM_calls','native_calls','manual_export_trace_calls'):eq(counts[name],0,name)
    if complete:
        for name in ('optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged','context_and_normalization_reused'):require(report[name],name)
        values=dict(condition='causal',ordinary_start_step=81000,ordinary_final_step=request['ordinary_final_step'],additional_updates=request['updates'],
            optimizer_start_step=16000,optimizer_step=request['optimizer_final_step'],fresh_optimizer=False,features=1323,context_features=323,
            head_output='normalized_target',fixed_full_state_coefficient=1.8188207859141674,execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',
            parity_tolerance_rad=1e-5,checkpoint_selection=False,native_steps=0,BFM_calls=0,training_first_layer_execution='split_old256_new256_original1000_plus323',
            hidden_width=512,architecture=[1323,512,512,23],export_first_layer_execution='monolithic_float64_1323',expansion_performed=False,
            recovery_rows=1018,recovery_phase_counts=[99,819,100],recovery_coefficient=request['recovery_coefficient'],
            source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],recovery_collection=request['subjects']['collection_report'])
        for name,value in values.items():eq(report.get(name),value,name)
        error=report['max_preclip_error_rad']
        if type(error) not in (int,float) or not math.isfinite(error) or error<0 or error>1e-5 or report['initial_restoration']['passed'] is not True:raise ValueError('Original numerical parity failed')
        eq(report['direct_subject_sha256'],{k:v['sha256'] for k,v in request['subjects'].items()},'all report consumed subjects')
    return complete

def validate_manifest(root,path):
    value=read(path)
    for name,digest in value['files'].items():
        p=(root/name).resolve()
        if p.parent!=root.resolve() or sha(p)!=digest:raise ValueError('Output manifest mismatch: '+str(p))
    return value

def static_pins(base,receipt,clear):
    result=dict(receipt['input_sha256'])
    for name,digest in receipt['source_sha256'].items():result[(Path(receipt['source_directory'])/name).as_posix()]=digest
    result[(base/'training_request.json').as_posix()]=clear['request_sha256']
    result[(base/'training_frozen_inputs.json').as_posix()]=clear['frozen_receipt_sha256']
    canonical_pins(result)
    return result

def validate_process(start,child,end,dispatch,request,clear,launch_sha,clear_sha,base):
    pids=[end['wrapper_pid'],end['child_pid']]
    if any(type(p)!=int or p<=0 for p in pids) or pids[0]==pids[1]:raise ValueError('Invalid actual process IDs')
    eq(start['wrapper_pid'],pids[0],'start PID');eq(child['wrapper_pid'],pids[0],'child parent PID');eq(child['child_pid'],pids[1],'child PID')
    require(child['captured_handle_nonzero'],'Child handle not captured')
    require(end['child_started'],'Child never started');require(end['exit_known'],'Unknown child exit')
    if type(end['raw_python_exit_code']) is not int or type(end['exit_code']) is not int:raise ValueError('Raw/diagnostic exits must be known integers')
    eq(dispatch['wrapper_pid'],pids[0],'dispatch PID');require(dispatch['captured_handle_nonzero'],'Wrapper handle not captured')
    expected_args=['-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(base/'run_fit_durable_v1.ps1'),'-ClearanceSha256',clear_sha,'-LaunchReceiptSha256',launch_sha]
    def normargs(args):return [str(v).replace('\\','/') for v in args]
    eq(normargs(dispatch['arguments']),normargs(expected_args),'exact wrapper arguments')
    eq(dispatch['executable'].replace('\\','/').casefold(),'c:/windows/system32/windowspowershell/v1.0/powershell.exe','PS5.1 executable')
    for item in (start,end,dispatch):
        eq(item['clearance_sha256'],clear_sha,'clearance SHA');eq(item['launch_receipt_sha256'],launch_sha,'launch SHA')
    for key in ('request_sha256','frozen_receipt_sha256'):eq(start[key],clear[key],key)
    for key in ('ordinary_start_step','ordinary_final_step','optimizer_start_step','optimizer_final_step','condition','updates','budgets','automatic_retry'):eq(start[key],request[key],key)
    command=start['command'];snapshot=base/'source_snapshot_v1'
    eq(Path(command['python']).resolve(),Path(request['runtime']['python_path']).resolve(),'actual runtime')
    eq([str(v).replace('\\','/') for v in command['arguments']],['-u',(snapshot/'train_recovery.py').as_posix()],'driver arguments')
    eq(Path(command['working_directory']).resolve(),snapshot.resolve(),'driver working directory')
    eq(command['CUBLAS_WORKSPACE_CONFIG'],':4096:8','CUBLAS');eq(command['threads'],1,'threads')
    return pids

def main():
    fit=BASE/'fit';process=BASE/'fit_process_v1'
    result=dict(owner_verification_passed=False,accounting_passed=False,completion_passed=False,numerical_completion_passed=False,
        task_model_calls=0,optimizer_updates=0,native_steps=0,behavioral_qualification=False)
    try:
        end=read(process/'exit.json');start=read(process/'start.json');child=read(process/'child.json');dispatch=read(process/'dispatch.json');attempt=read(process/'dispatch_attempt.json')
        for key in ('dispatcher_pid','executable','arguments','clearance_sha256','launch_receipt_sha256','automatic_retry'):eq(attempt[key],dispatch[key],'dispatch attempt '+key)
        receipt=read(BASE/'training_frozen_inputs.json');request=read(BASE/'training_request.json');clear=read(BASE/'training_clearance.json');launch=read(BASE/'launch_receipt.json')
        clear_sha=sha(BASE/'training_clearance.json');launch_sha=sha(BASE/'launch_receipt.json')
        require(clear['approved'],'Concrete clearance absent');eq(clear['automatic_retry'],False,'retry')
        for key in ('updates','ordinary_start_step','ordinary_final_step','optimizer_start_step','optimizer_final_step','condition'):eq(clear[key],request[key],key)
        for name,key in [('training_request.json','request_sha256'),('training_frozen_inputs.json','frozen_receipt_sha256')]:eq(sha(BASE/name),clear[key],key);eq(launch[key],clear[key],key)
        eq(receipt['training_request_sha256'],clear['request_sha256'],'frozen request')
        eq(Path(receipt['source_directory']).resolve(),(BASE/'source_snapshot_v1').resolve(),'snapshot directory')
        eq(clear['launch_receipt_sha256'],launch_sha,'clear launch');eq(sha(clear['launcher_path']),clear['launcher_sha256'],'launcher');eq(launch['launcher_sha256'],clear['launcher_sha256'],'launch launcher')
        eq(Path(clear['launcher_path']).resolve(),(BASE/'run_fit_durable_v1.ps1').resolve(),'literal launcher path')
        eq(sha(clear['review_path']),clear['review_sha256'],'concrete review');review=read(clear['review_path'])
        require(review['prelaunch_review_pass'],'Concrete review false')
        for key in ('frozen_receipt_sha256','launcher_sha256','launch_receipt_sha256'):eq(review[key],clear[key],'review '+key)
        eq(review['training_request_sha256'],clear['request_sha256'],'review request')
        expected=static_pins(BASE,receipt,clear)
        eq(canonical_pins(launch['input_sha256']),canonical_pins(expected),'full static launch membership');eq(launch['pin_count'],len(expected),'launch count')
        launch_pins=dict(expected)
        for path,digest in [(BASE/'launch_receipt.json',launch_sha),(BASE/'training_clearance.json',clear_sha),(Path(clear['review_path']),clear['review_sha256'])]:launch_pins[path.as_posix()]=digest
        for name in ('prerun_pins.json','postrun_pins.json'):validate_process_pin_report(read(process/name),launch_pins)
        require(end['all_postrun_pins_exact'],'Post-run pins false')
        pids=validate_process(start,child,end,dispatch,request,clear,launch_sha,clear_sha,BASE)
        observations=[absent(pid) for pid in pids]
        require(all(observations),'Observed process still present')
        absence=dict(wrapper_pid=pids[0],child_pid=pids[1],wrapper_absent=observations[0],child_absent=observations[1],
            dispatch_sha256=sha(process/'dispatch.json'),start_sha256=sha(process/'start.json'),child_sha256=sha(process/'child.json'),exit_sha256=sha(process/'exit.json'))
        write(process/'process_absence.json',absence)
        report=read(fit/'report.json');complete=validate_report(report,request);shared=fit/'shared'
        paths=dict(fit_report=fit/'report.json',training_manifest=BASE/'training_frozen_inputs.json',training_request=BASE/'training_request.json',
            process_exit=process/'exit.json',source_checkpoint=request['subjects']['checkpoint']['path'],source_fit_report=request['subjects']['fit_report']['path'],
            coefficient=request['subjects']['coefficient_source']['path'],energy_source=request['subjects']['energy_source']['path'],balance_source_review=request['subjects']['balance_source_review']['path'],
            full_state_generation_request=request['full_state_paths']['request'],full_state_generation_report=request['full_state_paths']['report'],
            full_state_data_audit=request['subjects']['full_state_root_audit']['path'],full_state_data_owner=request['subjects']['full_state_root_owner']['path'])
        for name in ('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','consistency_report','warm_restore_review','selected_protocol'):paths[name]=request['subjects'][name]['path']
        optional=dict(checkpoint=fit/'student_head.pt',head=fit/'student_head.onnx',export_manifest=fit/'output_manifest.json',normalization=shared/'normalization.npz',shared_manifest=shared/'output_manifest.json',context_alignment=shared/'context_alignment.json',recovery_evidence=shared/'recovery_evidence.json')
        for key,path in optional.items():
            if path.exists():paths[key]=path
        direct={key:sha(path) for key,path in paths.items()}
        if (shared/'output_manifest.json').exists():validate_manifest(shared,shared/'output_manifest.json')
        if complete:
            for key,field in [('training_request','training_request_sha256'),('training_manifest','frozen_receipt_sha256'),('checkpoint','checkpoint_sha256'),('head','onnx_sha256'),('normalization','normalization_sha256'),('export_manifest','output_manifest_sha256'),('shared_manifest','shared_manifest_sha256')]:eq(direct[key],report[field],'output '+key)
            validate_manifest(fit,fit/'output_manifest.json')
            eq(end['raw_python_exit_code'],0,'completed raw');eq(end['exit_code'],0,'completed wrapper');eq(end['error'],None,'completed error')
        else:
            failures=[p for p in (fit/'failure.json',fit/'setup_failure.json') if p.exists()]
            if not failures or end['raw_python_exit_code']==0 or end['exit_code']==0:raise ValueError('Untruthful failed fit verdict')
            for path in failures:eq(read(path)['automatic_retry'],False,'failed retry');direct[path.stem]=sha(path)
        outputs={str(path):sha(path) for path in [process/name for name in ('dispatch_attempt.json','dispatch.json','start.json','child.json','exit.json','prerun_pins.json','postrun_pins.json','process_absence.json','stdout.log','stderr.log','wrapper_stdout.log','wrapper_stderr.log')]}
        outputs.update({str(path):direct[key] for key,path in paths.items()})
        result.update(owner_verification_passed=True,accounting_passed=True,completion_passed=complete,numerical_completion_passed=complete,
            optimization_completed=report.get('optimization_completed') is True,direct_subject_sha256=direct,
            direct_subjects={k:dict(path=str(v),sha256=direct[k]) for k,v in paths.items()},output_sha256=outputs,
            expected_pids=pids,processes_absent=True,process_absence=absence,process_absence_sha256=sha(process/'process_absence.json'),
            raw_exit_known=True,raw_python_exit_code=end['raw_python_exit_code'],exit_code=end['exit_code'],all_postrun_pins_exact=True,
            current_pin_count=len(receipt['input_sha256'])+len(receipt['source_sha256']),launch_pin_count=len(launch_pins),
            launch_receipt_sha256=launch_sha,clearance_sha256=clear_sha,condition='causal')
    except BaseException as exc:result['error']=repr(exc)
    owner=BASE/'owner_completion_verification.json';write(owner,result)
    print(json.dumps(dict(accounting_passed=result['accounting_passed'],completion_passed=result['completion_passed'],sha256=sha(owner))))
    if not result['accounting_passed']:raise SystemExit(1)
if __name__=='__main__':main()
