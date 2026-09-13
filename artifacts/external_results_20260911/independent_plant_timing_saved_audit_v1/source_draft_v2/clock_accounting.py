"""Saved counters, raw fault capsules and observed process/owner receipts."""
import hashlib
from pathlib import Path
import numpy as np
from clock_saved_math import exact,repeated_time
from qualified_binary_math import typed_json
import clock_saved_math


def counts(a,metadata,report,capsules,timing_handoff=None):
    core=report['session']['foundation'];native=report['session']['stepper'];api=report['api_counters']
    saved_core=metadata['summary']['foundation'];n=len(a['step_index']);c=len(a['control_control'])
    for key in ('requested','attempted','returned','captured','verified','committed','unexecuted','history_entries','epoch_ns',
                'failure','input_fault','first_deadline_failure','deadline_misses','max_debt','cumulative_wake_debt'):
        if core[key]!=saved_core[key]:raise AssertionError('Stable foundation field changed after evidence export: '+key)
    for block,keys in [(core,('attempted','returned','captured','verified','committed')), (native,('attempted','returned','capture_attempts','captured','verification_attempts','verified')),
                       (api,('step_attempted','step_returned','serialization_attempted','serialization_returned','denied_step_calls','denied_serialization_calls'))]:
        if any(type(block[k]) is not int or block[k]<0 for k in keys):raise AssertionError('Nonnegative literal counter required')
    if not 0<=core['verified']<=core['captured']<=core['returned']<=core['attempted']<=18190:raise AssertionError('Foundation counter order')
    if not 0<=native['verified']<=native['verification_attempts']<=native['captured']<=native['capture_attempts']<=native['returned']<=native['attempted']<=18190:
        raise AssertionError('Native attempt/return/capture/verification counter order')
    if core['requested']!=18190 or core['unexecuted']!=18190-core['returned'] or core['committed']!=n:raise AssertionError('Full original scope or committed count')
    if core['returned']-n not in (0,1) or core['attempted']-core['returned'] not in (0,1):raise AssertionError('More than one failed outstanding native call')
    if not n<=core['captured']<=n+1 or not int(a['step_verified'].sum())<=core['verified']<=int(a['step_verified'].sum())+1:
        raise AssertionError('Captured/verified credit without saved or reserved last return')
    if core['captured']>n and 'last_validated_capture_state' not in capsules:raise AssertionError('Missing last uncommitted capture evidence')
    reserved=metadata.get('control_ledger_overflow')
    expected_history=c+int(reserved is not None)
    if core['history_entries']!=expected_history:raise AssertionError('History advanced without committed or reserved boundary')
    api_equal=api['step_attempted']==native['attempted'] and api['step_returned']==native['returned']
    owner_equal=all(native[k]==core[k] for k in ('returned','captured','verified'))
    if not (api_equal and owner_equal):
        if timing_handoff is None or timing_handoff.get('partial_counter_proof') is not True:
            raise AssertionError('Counter difference lacks independently checked terminal timing handoff')
    if not native['attempted']<=core['attempted']<=native['attempted']+1:raise AssertionError('Only pre-native validation may add one foundation attempt')
    expected=float(metadata['initial_time'])
    for _ in range(native['returned']):expected+=.002
    if native['expected_time']!=expected:raise AssertionError('Native independent expected clock changed')
    if (core['returned']<18190 or core['attempted']!=core['returned']) and core['failure'] is None and report['first_error'] is None and report['session']['driver_failure'] is None:
        raise AssertionError('Incomplete original scope lacks failure reason')
    fault_summary=None
    if 'native_fault_evidence' in capsules:
        fault=typed_json(capsules['native_fault_evidence'])
        for key in ('attempted','returned','captured'):
            if fault[key]!=native[key]:raise AssertionError('Raw native fault counter differs')
        if fault.get('integration_get_returned') is True:
            state=fault['integration']
            for key,section in [('qpos',slice(1,31)),('qvel',slice(31,60)),('ctrl',slice(89,112))]:
                if key in fault:exact(state[section],fault[key],'Fault full291/'+key+' overlap')
                elif key not in fault['field_errors']:raise AssertionError('Missing fault field lacks read-failure record')
            if 'time' in fault:exact(state[0],np.asarray(fault['time'],np.float64),'Fault time overlap')
            elif 'time' not in fault['field_errors']:raise AssertionError('Missing fault time lacks read-failure record')
        complete_capture=n>0 and native['attempted']==native['returned']==n and core['captured']==n
        if complete_capture and fault.get('integration_get_returned') is True and not fault['field_errors']:
            exact(fault['integration'],a['step_integration'][-1],'Fault actual last full291')
            exact(fault['qfrc_actuator'][6:],a['step_actuator_force'][-1],'Fault actual force')
            exact(fault['warnings'],a['step_warning_counts'][-1],'Fault warning counts')
            exact(fault['lastinfo'],a['step_warning_lastinfo'][-1],'Fault warning lastinfo')
        fault_summary={'typed_fault_present':True,'complete_capture_link_checked':bool(complete_capture),
                       'mutation_uncertain':native['attempted']>native['returned'],'field_errors':fault['field_errors']}
    for name in ('native_capture_return_evidence','last_foundation_capture_return'):
        if name in capsules:
            returned=typed_json(capsules[name])
            if isinstance(returned,dict) and returned.get('__type__')=='clock_core.CapturedStep' and n and core['captured']==n:
                if returned['state']!=a['packed_capture'][-1].tobytes() or returned['torque']!=a['step_command'][-1].tobytes():raise AssertionError('Raw returned capture differs')
    return {'timing_counter_handoff':timing_handoff,'foundation':{k:core[k] for k in ('attempted','returned','captured','verified','committed','unexecuted')},
            'reserved_uncommitted_history_entries':int(reserved is not None),
            'actual_native_attempted':api['step_attempted'],'actual_native_returned':api['step_returned'],
            'raw_fault':fault_summary,'complete_native_coverage':core['verified']==core['captured']==core['returned']==core['attempted']==native['capture_attempts']==native['verification_attempts']==18190 and api['denied_step_calls']==api['denied_serialization_calls']==0}


def owner_checks(owner,receipt,clearance,start,child,raw,exit_record,diagnostic,absence,pre,post,report):
    if owner['accounting_passed'] is not True or owner['pre_and_post_input_hashes_exact'] is not True:raise AssertionError('Owner accounting failed')
    if owner['input_hashes']!=receipt['input_hashes']:raise AssertionError('Owner launch input map differs')
    for record in (pre,post):
        if record['all_exact'] is not True or set(record['files'])!=set(receipt['input_hashes']):raise AssertionError('Launch pre/post pin coverage')
        for path,digest in receipt['input_hashes'].items():
            if record['files'][path]!={'expected':digest,'actual':digest,'matched':True}:raise AssertionError('Launch pin evidence differs')
    if start['wrapper_pid']!=child['wrapper_pid'] or child['handle_acquired'] is not True:raise AssertionError('Actual Windows child linkage')
    if absence['windows_expected_pids']!=[start['wrapper_pid'],child['child_pid']]:raise AssertionError('Absence did not check actual process IDs')
    if absence['windows_present'] or absence['linux_present'] or absence['all_observed_pids_absent'] is not True:raise AssertionError('Observed task process not absent')
    if raw['known'] is not True or type(raw['raw_python_exit_code']) is not int or raw['raw_error'] is not None:raise AssertionError('Unknown raw native runner exit')
    if exit_record['known'] is not True or exit_record['raw_python_exit_code']!=raw['raw_python_exit_code']:raise AssertionError('Raw exit not preserved')
    if exit_record['error'] is not None or exit_record['all_postrun_hashes_exact'] is not True:raise AssertionError('Wrapper preservation/exit failure')
    expected=0 if diagnostic['diagnostic_passed'] else 2
    if exit_record['diagnostic_exit_code']!=expected or exit_record['exit_code']!=expected:raise AssertionError('Diagnostic status differs')
    if report['component_preliminary_pass'] is not diagnostic['diagnostic_passed'] or owner['diagnostic_passed'] is not diagnostic['diagnostic_passed']:
        raise AssertionError('Owner diagnostic changed producer outcome')
    cleanup=report['worker_cleanup'] or {};uncertain=cleanup.get('start_side_effects_uncertain',False)
    if owner['process_absence_proven'] is (not uncertain) and owner['passed'] is (not uncertain):pass
    else:raise AssertionError('Failed-start uncertainty incorrectly removed')
    return {'accounting_passed':True,'observed_processes_absent':True,'unobserved_start_side_effects_uncertain':uncertain,
            'raw_python_exit_code':raw['raw_python_exit_code'],'diagnostic_passed':diagnostic['diagnostic_passed']}


def saved_timing_counters(a,report,timing):
    core=report['session']['foundation'];n=len(a['step_index']);epoch=core['epoch_ns']
    finish_debt=np.maximum(0,np.minimum(18190,(a['step_actual_end']-epoch)//2000000)-np.arange(1,n+1))
    committed_max=max([0]+a['step_wake_debt'].tolist()+finish_debt.tolist())
    wake_sum=int(a['step_wake_debt'].sum());extra=core['cumulative_wake_debt']-wake_sum
    all_cycles_committed=not timing['final_uncommitted_cycle_indices'] and core['attempted']==core['returned']==n
    if core['max_debt']<committed_max or extra<0:raise AssertionError('Saved clock debt counter below actual records')
    if all_cycles_committed and (core['max_debt']!=committed_max or extra!=0 or core['deadline_misses']!=len(timing['captured_step_deadline_misses'])):
        raise AssertionError('Complete captured debt/deadline accounting differs')
    if report['session']['outer_deadline_miss_indices']!=timing['outer_deadline_misses']:raise AssertionError('Outer deadline ledger summary differs')
    if n and np.any(a['step_wake_debt']>100):raise AssertionError('A native step exceeded fixed debt abort cap')
    if n and np.any(a['step_actual_start']-epoch>60000000000):raise AssertionError('A native step exceeded fixed elapsed cap')
    return dict(committed_max_debt=committed_max,committed_wake_debt_sum=wake_sum,uncommitted_wake_debt=extra,
                complete_counter_reconstruction=all_cycles_committed,
                limitation=None if all_cycles_committed else 'An uncommitted failed cycle may lack a start/finish sample; its retained aggregate debt gets no full timing credit.')


def uncommitted_capture(a,metadata,report,capsules,initial,contract):
    """Check the final returned/captured step even if its StepRecord could not commit."""
    core=report['session']['foundation'];n=len(a['step_index'])
    if core['captured']==n:return None
    if core['captured']!=core['returned'] or core['returned']!=n+1:raise AssertionError('Reserved capture count')
    returned=typed_json(capsules['last_foundation_capture_return'])
    if returned.get('__type__')!='clock_core.CapturedStep' or returned['warnings'].get('__type__')!='clock_core.WarningLedger':
        raise AssertionError('Reserved capture type')
    state=np.frombuffer(capsules['last_validated_capture_state'],np.float64)
    command=np.frombuffer(capsules['last_validated_capture_torque'],np.float64)
    if state.shape!=(373,) or command.shape!=(23,) or returned['state']!=state.tobytes() or returned['torque']!=command.tobytes():
        raise AssertionError('Reserved capture owned bytes differ')
    b={k:v.copy() for k,v in a.items()}
    fields=dict(packed_capture=state,step_integration=state[:291],step_qpos=state[291:321],step_qvel=state[321:350],step_actuator_force=state[350:],
        step_command=command,step_target=a['control_target'][n//10],step_raw_action=a['control_raw_action'][n//10],
        step_warning_counts=np.asarray(returned['warnings']['counts'],np.int32),step_warning_lastinfo=np.asarray(returned['warnings']['lastinfo'],np.int32),
        step_expected_simulation_time=np.asarray(returned['simulation_time'],np.float64),
        step_verified=np.asarray(core['verified']-int(a['step_verified'].sum())==1,np.bool_))
    for key,value in fields.items():b[key]=np.concatenate([b[key],value[None]])
    verified=clock_saved_math.physical_arrays(b,initial,contract)
    return {'actual_reserved_native_index':n,'actual_full291_checked':True,'strict_capture_passed':bool(b['step_verified'][-1]),
            'committed_step_credit':0,'full_scope_credit':False,'strict_failures':verified['strict_failures']}
