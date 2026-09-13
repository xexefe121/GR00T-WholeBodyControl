"""Source-only adaptation of previously reviewed durable process helpers."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_context_study_v2'
text=(OLD/'run_fit_durable_v2.ps1').read_text(encoding='utf-8')
def change(old,new,count=1):
    global text
    if text.count(old)!=count:raise ValueError('Unexpected wrapper derivation: '+old[:90])
    text=text.replace(old,new)
change("'fit_process_v2'","'fit_process_v1'")
change("$clear.updates_per_condition -ne 3000 -or $clear.ordinary_final_step -ne 68000 -or (@($clear.conditions) -join ',') -ne 'blinded,causal'",
       "$clear.updates -ne 3000 -or $clear.ordinary_final_step -ne 71000 -or $clear.condition -ne 'causal' -or $clear.optimizer_start_step -ne 3000 -or $clear.optimizer_final_step -ne 6000")
change("$request.kind -ne 'matched_causal_context_utility_study' -or $request.root_selected -ne $true -or $request.updates_per_condition -ne 3000 -or (@($request.conditions) -join ',') -ne 'blinded,causal'",
       "$request.kind -ne 'causal_response_balanced_continuation' -or $request.root_selected -ne $true -or $request.updates -ne 3000 -or $request.condition -ne 'causal' -or $request.fresh_optimizer -ne $false -or $request.ordinary_final_step -ne 71000")
change("'train_context_pair.py'","'train_response_balanced.py'")
change("kind='matched_causal_context_utility_study'","kind='causal_response_balanced_continuation'")
change("updates_per_condition=3000;ordinary_final_step=68000;conditions=@('blinded','causal');budgets=$request.budgets;automatic_retry=$false",
       "updates=3000;ordinary_start_step=68000;ordinary_final_step=71000;condition='causal';optimizer_start_step=3000;optimizer_final_step=6000;budgets=$request.budgets;automatic_retry=$false")
start=text.index("    $pair=Read-SharedJson")
end=text.index('    $runExit=0',start)
text=text[:start]+'''    $fitRoot=Join-Path $runRoot 'fit'
    $report=Read-SharedJson (Join-Path $fitRoot 'report.json')
    foreach($field in @('completed','optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged')){if($report.$field -ne $true){throw "Incomplete fit report: $field"}}
    if($report.condition -ne 'causal' -or $report.ordinary_start_step -ne 68000 -or $report.ordinary_final_step -ne 71000 -or $report.additional_updates -ne 3000 -or $report.optimizer_start_step -ne 3000 -or $report.optimizer_step -ne 6000 -or $report.fresh_optimizer -ne $false -or $report.features -ne 1323 -or $report.head_output -ne 'normalized_target'){throw 'Final warm continuation metadata differs.'}
    if($report.fixed_full_state_coefficient -ne 1.8188207859141674d -or $report.parity_tolerance_rad -ne 0.00001d -or $report.max_preclip_error_rad -gt 0.00001d -or $report.initial_restoration.passed -ne $true){throw 'Original fixed numerical gate differs.'}
    if($report.counts.training.rows_verified -ne 44058000 -or $report.counts.training.calls_verified -ne 9000 -or $report.counts.calibration_forward_calls -ne 0 -or $report.counts.calibration_gradient_calls -ne 0 -or $report.counts.BFM_calls -ne 0 -or $report.counts.native_calls -ne 0){throw 'Training counts differ.'}
    foreach($backend in @('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')){if($report.counts.diagnostics.$backend.rows_verified -ne 367570 -or $report.counts.diagnostics.$backend.calls_verified -ne 1437){throw "Incomplete backend counts: $backend"}}
    if((Read-SharedSha (Join-Path $fitRoot 'student_head.pt')) -ne $report.checkpoint_sha256 -or (Read-SharedSha (Join-Path $fitRoot 'student_head.onnx')) -ne $report.onnx_sha256){throw 'Output identities differ.'}
'''+text[end:]
change("kind='matched_causal_context_utility_study_exit'","kind='causal_response_balanced_continuation_exit'")
(BASE/'run_fit_durable_v1.ps1').write_text(text,encoding='utf-8',newline='\n')
reader=(OLD/'read_fit_progress_v2.ps1').read_text(encoding='utf-8')
left=reader.index("foreach($name in @(");right=reader.index(')){',left)
reader=reader[:left]+"foreach($name in @('fit_process_v1/start.json','fit_process_v1/child.json','fit/progress.json','fit/failure.json','fit/setup_failure.json','fit/report.json','fit_process_v1/exit.json'"+reader[right:]
(BASE/'read_fit_progress_v1.ps1').write_text(reader,encoding='utf-8',newline='\n')

# Reuse exact strict pin, process identity, counter and manifest primitives.
owner=(OLD/'verify_completed_v3.py').read_text(encoding='utf-8')
start=owner.index('def condition_subjects(');end=owner.index('def absent(',start)
owner=owner[:start]+owner[end:]
start=owner.index('def validate_report(');end=owner.index('def validate_manifest(',start)
owner=owner[:start]+'''def validate_report(report):
    complete=report.get('completed') is True
    if 'counts' not in report:
        if complete or report.get('task_forward_calls')!=0 or report.get('optimizer_updates')!=0:
            raise ValueError('Missing truthful setup-failure counters.')
        return False
    counts=report['counts']
    validate_counter(counts['training'],9000,44058000,complete)
    for backend in BACKENDS:validate_counter(counts['diagnostics'][backend],1437,367570,complete)
    for name in ('calibration_forward_calls','calibration_gradient_calls','BFM_calls','native_calls','manual_export_trace_calls'):
        if counts[name]!=0:raise ValueError('Unexpected calls: '+name)
    if complete:
        for name in ('optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged','context_and_normalization_reused'):
            if report[name] is not True:raise ValueError('Inconsistent completed fit: '+name)
        values=dict(condition='causal',ordinary_start_step=68000,ordinary_final_step=71000,additional_updates=3000,
            optimizer_start_step=3000,optimizer_step=6000,fresh_optimizer=False,features=1323,context_features=323,
            head_output='normalized_target',fixed_full_state_coefficient=1.8188207859141674,
            execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',parity_tolerance_rad=1e-5,
            checkpoint_selection=False,native_steps=0,BFM_calls=0,
            training_first_layer_execution='split_contiguous_1000_plus_323',export_first_layer_execution='monolithic_float64_1323')
        for name,value in values.items():
            if report.get(name)!=value:raise ValueError('Warm continuation contract differs: '+name)
        error=report['max_preclip_error_rad']
        if not math.isfinite(error) or error<0 or error>1e-5 or report['initial_restoration']['passed'] is not True:
            raise ValueError('Numerical parity failed.')
    return complete

'''+owner[end:]
owner=owner[:owner.index('def main():')]+'''def main():
    fit=BASE/'fit';process=BASE/'fit_process_v1'
    result=dict(owner_verification_passed=False,accounting_passed=False,completion_passed=False,numerical_completion_passed=False,
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
        for path,digest in pins.items():
            if sha(path)!=digest:raise ValueError('Current frozen pin differs: '+path)
        for name,key in [('training_request.json','request_sha256'),('training_frozen_inputs.json','frozen_receipt_sha256')]:
            if sha(BASE/name)!=clear[key] or start[key]!=clear[key]:raise ValueError('Clearance/request identity differs.')
        if sha(BASE/'training_clearance.json')!=start['clearance_sha256'] or exit_report['clearance_sha256']!=start['clearance_sha256']:raise ValueError('Clearance changed.')
        if sha(clear['review_path'])!=clear['review_sha256'] or sha(clear['launcher_path'])!=clear['launcher_sha256']:raise ValueError('Concrete release changed.')
        launch_pins=dict(pins)
        for name,digest in [('training_request.json',clear['request_sha256']),('training_frozen_inputs.json',clear['frozen_receipt_sha256']),('training_clearance.json',start['clearance_sha256'])]:launch_pins[(BASE/name).as_posix()]=digest
        launch_pins[clear['review_path']]=clear['review_sha256']
        for name in ('prerun_pins.json','postrun_pins.json'):validate_process_pin_report(read(process/name),launch_pins)
        report=read(fit/'report.json');complete=validate_report(report)
        shared=fit/'shared'
        paths=dict(fit_report=fit/'report.json',training_manifest=BASE/'training_frozen_inputs.json',
            training_request=BASE/'training_request.json',process_exit=process/'exit.json',
            source_checkpoint=request['subjects']['checkpoint']['path'],source_fit_report=request['subjects']['fit_report']['path'],
            coefficient=request['subjects']['coefficient_source']['path'],energy_source=request['subjects']['energy_source']['path'],
            balance_source_review=request['subjects']['balance_source_review']['path'],
            full_state_generation_request=request['full_state_paths']['request'],full_state_generation_report=request['full_state_paths']['report'],
            full_state_data_audit=request['subjects']['full_state_root_audit']['path'],full_state_data_owner=request['subjects']['full_state_root_owner']['path'])
        optional=dict(checkpoint=fit/'student_head.pt',head=fit/'student_head.onnx',export_manifest=fit/'output_manifest.json',
            normalization=shared/'normalization.npz',shared_manifest=shared/'output_manifest.json',context_alignment=shared/'context_alignment.json')
        for key,path in optional.items():
            if path.exists():paths[key]=path
        direct={key:sha(path) for key,path in paths.items()}
        if (shared/'output_manifest.json').exists():validate_manifest(shared,shared/'output_manifest.json')
        if complete:
            if report['training_request_sha256']!=direct['training_request'] or report['frozen_receipt_sha256']!=direct['training_manifest']:raise ValueError('Fit input bindings differ.')
            for key,field in [('checkpoint','checkpoint_sha256'),('head','onnx_sha256'),('normalization','normalization_sha256'),('export_manifest','output_manifest_sha256'),('shared_manifest','shared_manifest_sha256')]:
                if direct[key]!=report[field]:raise ValueError('Fit output identity differs: '+key)
            validate_manifest(fit,fit/'output_manifest.json')
            if exit_report['raw_python_exit_code']!=0 or exit_report['exit_code']!=0 or exit_report['error'] is not None:raise ValueError('Completed fit exit failed.')
        else:
            failures=[path for path in (fit/'failure.json',fit/'setup_failure.json') if path.exists()]
            if not failures or exit_report['raw_python_exit_code']==0 or exit_report['exit_code']==0:raise ValueError('Untruthful failed-fit verdict.')
            for path in failures:
                record=read(path)
                if record['automatic_retry'] is not False:raise ValueError('Failed attempt permits retry.')
                direct[path.stem]=sha(path)
        result.update(owner_verification_passed=True,accounting_passed=True,completion_passed=complete,
            numerical_completion_passed=complete,optimization_completed=report.get('optimization_completed') is True,
            direct_subject_sha256=direct,expected_pids=pids,processes_absent=True,raw_exit_known=known,
            raw_python_exit_code=exit_report['raw_python_exit_code'],exit_code=exit_report['exit_code'],
            all_postrun_pins_exact=True,current_pin_count=len(pins),launch_pin_count=len(launch_pins),condition='causal')
    except BaseException as exc:result['error']=repr(exc)
    owner=BASE/'owner_completion_verification.json'
    with owner.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\\n')
    print(json.dumps(dict(accounting_passed=result['accounting_passed'],completion_passed=result['completion_passed'],sha256=sha(owner))))
    if not result['accounting_passed']:raise SystemExit(1)
if __name__=='__main__':main()
'''
owner=owner.replace('"""Saved-only pair accounting; numerical completion is a separate verdict."""','"""Saved-only warm-fit accounting; numerical completion remains separate."""').replace("CONDITIONS=('blinded','causal')\n",'')
(BASE/'verify_completed.py').write_text(owner,encoding='utf-8',newline='\n')
print('Durable launcher, shared-delete monitor and single-fit owner checker written. No dispatch.')
