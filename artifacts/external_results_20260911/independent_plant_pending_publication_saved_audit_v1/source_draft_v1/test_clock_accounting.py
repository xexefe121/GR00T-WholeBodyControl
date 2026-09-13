"""Synthetic preservation/accounting corruption tests; zero external task calls."""
import base64,copy,hashlib,json,unittest
import numpy as np
from clock_accounting import counts,owner_checks,uncommitted_capture
from clock_saved_math import timing_arrays
from test_clock_saved_math import fixture
from clock_control_math import reserved_control
from test_clock_protocol_math import small_case


def typed(value):
    if type(value) is bytes:return dict(type='builtins.bytes',size=len(value),prefix=base64.b64encode(value).decode(),sha256=hashlib.sha256(value).hexdigest())
    if type(value) is float:return dict(type='builtins.float',hex=value.hex())
    if isinstance(value,np.ndarray):return dict(type='numpy.ndarray',dtype=value.dtype.str,shape=list(value.shape),data=typed(value.tobytes()))
    if type(value) is dict:return dict(type='builtins.dict',size=len(value),items=[[typed(k),typed(v)] for k,v in value.items()])
    if type(value) is tuple:return dict(type='builtins.tuple',size=len(value),items=[[typed(i),typed(v)] for i,v in enumerate(value)])
    return dict(type='builtins.'+type(value).__name__,value=value)


def count_case(n=2):
    a,initial,contract,outer,epoch=fixture(n);a['control_control']=np.array([0],np.int64)
    a['control_target']=np.zeros((1,23));a['control_raw_action']=np.zeros((1,23),np.float32)
    core=dict(requested=18190,attempted=n,returned=n,captured=n,verified=n,committed=n,unexecuted=18190-n,history_entries=1,
        epoch_ns=epoch,failure=dict(reason='SYNTHETIC_STOP'),input_fault=None,first_deadline_failure=None,deadline_misses=0,max_debt=0,cumulative_wake_debt=0)
    native=dict(attempted=n,returned=n,capture_attempts=n,captured=n,verification_attempts=n,verified=n,expected_time=float(a['step_integration'][-1,0]),initial_time=0.)
    api=dict(step_attempted=n,step_returned=n,serialization_attempted=4,serialization_returned=4,denied_step_calls=0,denied_serialization_calls=0)
    report=dict(session=dict(foundation=core,stepper=native,driver_failure=None),api_counters=api,first_error=None)
    metadata=dict(summary=copy.deepcopy(report['session']),initial_time=0.)
    return a,metadata,report,{},initial,contract,outer,epoch


def owner_case():
    pins={'synthetic/file':'a'*64};checks=dict(all_exact=True,files={k:dict(expected=v,actual=v,matched=True) for k,v in pins.items()})
    owner=dict(accounting_passed=True,pre_and_post_input_hashes_exact=True,input_hashes=pins,diagnostic_passed=False,process_absence_proven=True,passed=True)
    receipt=dict(input_hashes=pins);start=dict(wrapper_pid=1);child=dict(wrapper_pid=1,child_pid=2,handle_acquired=True)
    raw=dict(known=True,raw_python_exit_code=0,raw_error=None)
    exit_record=dict(known=True,raw_python_exit_code=0,error=None,all_postrun_hashes_exact=True,diagnostic_exit_code=2,exit_code=2)
    absence=dict(windows_expected_pids=[1,2],windows_present=[],linux_present=[],all_observed_pids_absent=True)
    return owner,receipt,{},start,child,raw,exit_record,dict(diagnostic_passed=False),absence,checks,copy.deepcopy(checks),dict(component_preliminary_pass=False,worker_cleanup={})


class AccountingTests(unittest.TestCase):
    def test_honest_partial_counts_no_scope_credit(self):
        a,m,r,c,*_=count_case();result=counts(a,m,r,c);self.assertFalse(result['complete_native_coverage'])
    def test_counter_mutation_and_missing_history_rejected(self):
        for block,key in [('foundation','attempted'),('stepper','returned')]:
            a,m,r,c,*_=count_case();r['session'][block][key]+=1
            with self.assertRaises(AssertionError):counts(a,m,r,c)
        a,m,r,c,*_=count_case();r['session']['foundation']['history_entries']=2;m['summary']=copy.deepcopy(r['session'])
        with self.assertRaisesRegex(AssertionError,'History advanced'):counts(a,m,r,c)
    def test_raw_zero_worker_failure_is_not_diagnostic_pass(self):
        result=owner_checks(*owner_case());self.assertTrue(result['accounting_passed']);self.assertFalse(result['diagnostic_passed'])
    def test_wrong_pid_missing_pin_unknown_raw_exit_rejected(self):
        for change in ('pid','pin','exit'):
            args=list(copy.deepcopy(owner_case()))
            if change=='pid':args[8]['windows_expected_pids']=[1,99]
            if change=='pin':args[10]['files']={}
            if change=='exit':args[5]['known']=False
            with self.assertRaises(AssertionError):owner_checks(*args)
    def test_failed_start_uncertainty_remains(self):
        args=list(owner_case());args[-1]['worker_cleanup']={'start_side_effects_uncertain':True}
        args[0]['process_absence_proven']=False;args[0]['passed']=False
        self.assertTrue(owner_checks(*args)['unobserved_start_side_effects_uncertain'])
        args[0]['passed']=True
        with self.assertRaisesRegex(AssertionError,'uncertainty'):owner_checks(*args)
    def test_final_pre_epoch_uncaptured_cycle_allowed_no_timing_credit(self):
        a,_,_,_,epoch=fixture(0)
        outer=[dict(index=0,returned=0,cycle_return_ns=epoch-1,fixed_nominal_end_ns=epoch+2000000)]
        result=timing_arrays(a,outer,epoch,0)
        self.assertEqual(result['final_uncommitted_cycle_indices'],[0]);self.assertFalse(result['fixed_epoch_timing_pass'])
    def test_final_uncommitted_return_has_no_skip_or_full_credit(self):
        a,_,_,outer,epoch=fixture(2);outer.append(dict(index=2,returned=3,cycle_return_ns=epoch+7000000,fixed_nominal_end_ns=epoch+6000000))
        result=timing_arrays(a,outer,epoch,3);self.assertEqual(result['final_uncommitted_cycle_indices'],[2])
        outer[-1]['returned']=4
        with self.assertRaises(AssertionError):timing_arrays(a,outer,epoch,3)
    def test_reserved_capture_actual_bytes_checked_without_commit_credit(self):
        a,m,r,c,initial,contract,*_=count_case(2)
        state=a['packed_capture'][-1].copy();state[0]+=.002;command=a['step_command'][-1].copy()
        returned=dict(type='clock_core.CapturedStep',fields=dict(simulation_time=typed(float(state[0])),state=typed(state.tobytes()),torque=typed(command.tobytes()),
            warnings=dict(type='clock_core.WarningLedger',fields=dict(counts=typed((0,)*8),lastinfo=typed((0,)*8)))))
        c.update(last_foundation_capture_return=json.dumps(returned).encode(),last_validated_capture_state=state.tobytes(),last_validated_capture_torque=command.tobytes())
        for block in ('foundation','stepper'):
            for key in ('attempted','returned','captured','verified'):r['session'][block][key]=3
        r['session']['foundation']['unexecuted']=18187;r['session']['stepper']['expected_time']=float(state[0]);r['api_counters']['step_attempted']=3;r['api_counters']['step_returned']=3
        r['session']['stepper']['capture_attempts']=3;r['session']['stepper']['verification_attempts']=3
        m['summary']=copy.deepcopy(r['session'])
        self.assertEqual(counts(a,m,r,c)['actual_native_returned'],3)
        result=uncommitted_capture(a,m,r,c,initial,contract);self.assertEqual(result['committed_step_credit'],0)
        broken=state.copy();broken[89]=.1;c['last_validated_capture_state']=broken.tobytes()
        with self.assertRaisesRegex(AssertionError,'owned bytes'):uncommitted_capture(a,m,r,c,initial,contract)
    def test_native_fault_last_state_and_counter_mutations(self):
        a,m,r,c,*_=count_case();state=a['step_integration'][-1]
        fault=dict(attempted=2,returned=2,captured=2,integration_get_returned=True,integration=state,qpos=state[1:31],qvel=state[31:60],ctrl=state[89:112],time=float(state[0]),
            qfrc_actuator=np.zeros(29),warnings=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32),field_errors={})
        c['native_fault_evidence']=json.dumps(typed(fault)).encode();self.assertTrue(counts(a,m,r,c)['raw_fault']['complete_capture_link_checked'])
        fault['returned']=1;c['native_fault_evidence']=json.dumps(typed(fault)).encode()
        with self.assertRaisesRegex(AssertionError,'fault counter'):counts(a,m,r,c)
    def test_reserved_history_boundary_owns_evidence_not_control_credit(self):
        case=small_case();a,m=case[:2];initial=a['control_integration'][0].copy()
        row=dict(control=0,physics=0,nominal_activation_ns=100000000,actual_activation_ns=100000000,admitted_ns=99999900,
            snapshot_state=initial.tobytes(),incoming_raw=a['control_incoming_raw'][0].tobytes(),flat_history_before=a['control_flat_history_before'][0].tobytes(),
            held=False,history_before=m['control_history_before'][0],history_after=m['control_history_after'][0],terms=m['control_measured_terms'][0],
            nominal_window_id=m['control_nominal_windows'][0],active_window_id=m['control_active_windows'][0],
            command=dict(dataclass='Command',fields=dict(command_id=m['control_command_ids'][0],origin='synthetic',target=a['control_target'][0].tobytes(),raw_action=a['control_raw_action'][0].tobytes())))
        a={k:v[:0].copy() for k,v in a.items()};a.update(step_index=np.empty(0,np.int64),control_physics=np.empty(0,np.int64),control_nominal_activation_ns=np.empty(0,np.int64))
        for key in ('control_history_before','control_history_after','control_measured_terms','control_command_ids','control_nominal_windows','control_active_windows'):m[key]=[]
        m['control_ledger_overflow']=dict(dataclass='ControlRecord',fields=row)
        result=reserved_control(a,m,initial,dict(default_q=np.zeros(23)),case[6],case[7],100000000)
        self.assertEqual(result['committed_control_credit'],0);self.assertEqual(result['independently_checked_history_entries'],1)
        row['incoming_raw']=np.ones(23,np.float32).tobytes()
        with self.assertRaisesRegex(AssertionError,'Previous actually'):reserved_control(a,m,initial,dict(default_q=np.zeros(23)),case[6],case[7],100000000)


if __name__=='__main__':unittest.main()
