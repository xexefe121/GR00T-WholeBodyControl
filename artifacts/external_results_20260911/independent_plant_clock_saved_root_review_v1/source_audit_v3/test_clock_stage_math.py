"""Synthetic stage receipts only; no native calls or task arrays."""
import copy
from pathlib import Path
import numpy as np
import pytest
from clock_stage_math import stages,source_contract,request_contract,BUDGETS,NS,COUNTERS,NATIVE


def fixture(n=18190):
    chosen=130*NS;epoch=chosen+200000000;entered=epoch+36390000000
    setup={'stage':'setup','entered_ns':0,'deadline_ns':240*NS}
    plant={'stage':'plant','entered_ns':chosen+1000000,'deadline_ns':epoch+120*NS}
    preserve={'stage':'preservation','entered_ns':entered,'deadline_ns':entered+180*NS}
    def w(ts):return {'phase':ts[-1]['stage'],'deadline_ns':ts[-1]['deadline_ns'],'transitions':copy.deepcopy(ts),
        'timeouts':[],'budgets':dict(BUDGETS),'native_signal_delivery_may_be_delayed':True}
    def counts(n,serial,foundation=True):return {'foundation':{k:n for k in COUNTERS} if foundation else None,
        'adapter':dict({k:n for k in NATIVE},stage='READY'),
        'api':{'step_attempted':n,'step_returned':n,'serialization_attempted':serial,'serialization_returned':serial},
        'native_return_may_be_uncertain':False,'extra_capture_or_native_read':False}
    report={'epoch_ns':epoch,'epoch_chosen_ns':chosen,'setup_finished_ns':chosen+100000000,'epoch_rebased':False,
        'first_error':None,'runtime':{'pid':7},'session':{'foundation':{k:n for k in COUNTERS},'stepper':{k:n for k in NATIVE}},
        'api_counters':{'step_attempted':n,'step_returned':n,'serialization_attempted':4,'serialization_returned':4,
            'serialization_records':[{'native_returned':True} for _ in range(4)]},'stage_watchdog':w([setup,plant,preserve])}
    def record(name,time,ts,**kw):return dict(stage=name,record_created_ns=time,request_sha256='r',pid=7,watchdog=w(ts),**kw)
    records=[record('setup_started',1,[setup]),record('native_setup_started',10*NS,[setup]),
        record('native_restored',128*NS,[setup],counters=counts(0,2,False)),
        record('epoch_armed',chosen+50000000,[setup,plant],epoch_chosen_ns=chosen,epoch_ns=epoch,
            allocation_finished_ns=chosen+10000000,counters=counts(0,2),worker_pid=8),
        record('plant_or_setup_exit',entered+1000000,[setup,plant,preserve],first_error=None,epoch_chosen_ns=chosen,
            epoch_ns=epoch,setup_finished_ns=report['setup_finished_ns'],counters=counts(n,2)),
        record('preservation_complete',entered+20*NS,[setup,plant,preserve],counters=counts(n,4),report_sha256='report',output_manifest_sha256='manifest')]
    raw={'known':True,'raw_python_exit_code':0,'raw_error':None};arrays={'step_index':np.arange(n),'step_verified':np.ones(n,bool)}
    return records,report,raw,arrays


def check(f):return stages(f[0],f[1],f[2],'r','report','manifest',f[3])


def test_full_counter_stage_scope():
    result=check(fixture());assert result['stage_watchdog_qualified'] and result['counter_blocks_equal_final_native_ledger']


@pytest.mark.parametrize('raw',[124,137,2,None,False])
def test_hard_timeout_unknown_or_nonzero_never_qualifies(raw):
    f=fixture();f[2]['raw_python_exit_code']=raw
    result=check(f);assert result['stage_evidence_integrity_passed'] and not result['stage_watchdog_qualified']


def test_complete_partial_native_packet_gets_no_full_clock_credit():
    result=check(fixture(3));assert result['stage_evidence_integrity_passed'] and not result['stage_watchdog_qualified']


def test_missing_final_stage_has_no_qualification():
    f=fixture();f[0].pop();result=check(f)
    assert not result['preservation_complete'] and not result['stage_watchdog_qualified']


def test_interrupted_epoch_fsync_retains_unknown_setup_gate_without_credit():
    f=fixture(0);r,report,raw,a=f
    report['setup_finished_ns']=None;report['first_error']={'type':'StageTimeout'}
    r[4]['setup_finished_ns']=None;r[4]['first_error']=report['first_error'];raw['raw_python_exit_code']=2
    result=check(f);assert result['stage_evidence_integrity_passed'] and not result['stage_watchdog_qualified']


@pytest.mark.parametrize('mutation',['deadline','rearm','pid','epoch','pre_epoch_step','exit_count','final_count','report_subject','report_watchdog','record_clock','missing_epoch','serialization'])
def test_corrupted_durable_stage_evidence_rejected(mutation):
    f=fixture();r,report,raw,a=f
    if mutation=='deadline':r[3]['watchdog']['deadline_ns']+=1
    elif mutation=='rearm':r[3]['watchdog']['transitions'].append(copy.deepcopy(r[3]['watchdog']['transitions'][-1]))
    elif mutation=='pid':r[-1]['pid']=9
    elif mutation=='epoch':r[3]['epoch_ns']+=1
    elif mutation=='pre_epoch_step':r[3]['counters']['api']['step_returned']=1
    elif mutation=='exit_count':r[4]['counters']['foundation']['returned']-=1
    elif mutation=='final_count':r[5]['counters']['adapter']['captured']-=1
    elif mutation=='report_subject':r[-1]['report_sha256']='other'
    elif mutation=='report_watchdog':report['stage_watchdog']['deadline_ns']+=1
    elif mutation=='record_clock':r[-1]['record_created_ns']=True
    elif mutation=='missing_epoch':r.pop(3)
    else:r[4]['counters']['api']['serialization_attempted']=4
    with pytest.raises(AssertionError):check(f)


def test_fixed_budget_and_original_timeout_coverage_reconstruction():
    request={'watchdog_budgets':dict(BUDGETS),'epoch_lead_ns':200000000,'debt_abort_steps':100,'elapsed_abort_ns':60*NS,
        'native_step_budget':18190,'serialization_budget':4,'outer_process_timeout_seconds':555,'requested_controls':1819,
        'main_controls':1569,'hold_controls':250,'epoch_rebase_allowed':False}
    args=['timeout','--signal=TERM','--kill-after=5s','555s','env'];request_contract(request,args)
    with pytest.raises(AssertionError):request_contract(request,['timeout','--signal=TERM','--kill-after=5s','120s'])
    request['debt_abort_steps']=101
    with pytest.raises(AssertionError):request_contract(request,args)


def test_actual_source_only_hashes_and_hot_loop_AST():
    new=Path(__file__).resolve().parents[2]
    paths={'original_runner':new/'independent_plant_process_clock_v1/source_runner_v1/run_clock.py',
        'clock_runner':new/'independent_plant_clock_timeout_correction_v1/source_draft_v1/run_clock.py',
        'clock_watchdog':new/'independent_plant_clock_timeout_correction_v1/source_draft_v1/stage_watchdog.py'}
    result=source_contract(paths);assert result['original_hot_loop_AST_exact'] and result['producer_imports']==0
