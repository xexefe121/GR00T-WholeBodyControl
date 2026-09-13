"""Minimal original counterexamples and valid partial-tail controls; arrays only."""
import copy
import numpy as np
import pytest
from test_retry_publication_math import fixture
from retry_publication_math import check,digest,encoded


def extend(f,n):
    epoch=f[5]
    f[6]=dict(step_actual_start=epoch+np.arange(n)*2000000,
              step_actual_end=epoch+np.arange(n)*2000000+1000000)
    f[4]['returned']=n


def expire(f,physics):
    f[2].append(dict(reason='PENDING_JOB_EXPIRED',activation=1,physics=physics,
        expiration='ACTIVATION_BOUNDARY',observed_ns=120000000,attempts=1,last_attempt_index=0,
        last_status='BUSY',deadline_ns=120000000,payload_sha256=digest(f[1]['1'])))
    f[4]['pending_publication']=None;f[4]['pending_expirations']=1


def unknown_tail():
    f=fixture();f[2].pop();f[7]=[];last=f[4]['last_job_publication']
    last.update(last_status=None,event_recorded=False)
    f[4].update(failure={'reason':'transport exception','physics':1},returned=1,
        job_publication_returned=1,job_publication_events=1,pending_publication=copy.deepcopy(last))
    extend(f,1)
    return f


def test_original_future_expiry_reproduction_now_rejected():
    f=fixture(('BUSY',));f[4]['failure']={'reason':'native strict failure','physics':1}
    expire(f,10)
    with pytest.raises(AssertionError,match='beyond actual terminal'):check(*f)


def test_original_skipped_retry_expiry_now_rejected():
    f=fixture(('BUSY',));extend(f,11);f[4]['failure']={'reason':'strict failure','physics':11}
    expire(f,10)
    with pytest.raises(AssertionError,match='immediately next'):check(*f)


def test_original_pending_tail_gap_now_rejected():
    f=fixture(('BUSY',));extend(f,5);f[4]['failure']={'reason':'strict failure','physics':5}
    with pytest.raises(AssertionError,match='omitted completed'):check(*f)


def test_original_unknown_return_before_later_native_steps_rejected():
    f=unknown_tail();extend(f,4)
    with pytest.raises(AssertionError):check(*f)


def test_truthful_failed_before_next_retry_preserves_pending():
    f=fixture(('BUSY',));f[4]['failure']={'reason':'clock/debt fault before next retry','physics':1}
    result=check(*f);assert result['pending_present'] and result['terminal_tick_coverage_exact']


def test_truthful_busy_after_attempt_but_before_native_step():
    f=fixture(('BUSY','BUSY'));f[4]['failure']={'reason':'interruption before native','physics':1}
    extend(f,1)
    assert check(*f)['pending_present']


def test_truthful_busy_before_uncommitted_returned_native_step():
    f=fixture(('BUSY','BUSY'));f[4]['failure']={'reason':'capture/record exception','physics':2}
    extend(f,1);f[4]['returned']=2
    assert check(*f)['pending_present']


def test_truthful_unknown_terminal_write_has_no_returned_credit():
    f=unknown_tail();result=check(*f)
    assert result['returned']==1 and result['attempted']==2 and result['terminal_tick_coverage_exact']


def test_bound_outer_watchdog_interruption_without_core_failure():
    f=unknown_tail();f[4]['failure']=None
    result=check(*f,terminal_interruption={'type':'StageTimeout','detail':'plant watchdog deadline reached'})
    assert result['bound_outer_interruption_present'] and result['returned']==1
    with pytest.raises(AssertionError):check(*f)


def test_foundation_cannot_continue_after_unknown_publication():
    f=unknown_tail();f[2].append(dict(reason='MAILBOX_BUSY',physics=2,slot=0))
    with pytest.raises(AssertionError,match='continued'):check(*f)


def test_pending_job_cannot_survive_input_fault_latch():
    f=fixture(('BUSY',));f[4]['input_fault']={'reason':'COMMAND_DEADLINE_MISSED'}
    with pytest.raises(AssertionError,match='latched'):check(*f)


def test_unreturned_count_cannot_exceed_reserved_native_capture():
    f=unknown_tail();f[4]['returned']=3
    with pytest.raises(AssertionError,match='terminal tick coverage'):check(*f)


def test_other_transport_cannot_follow_unknown_write_same_tick():
    f=unknown_tail()
    f[3].append(dict(direction='plant-results',operation='poll',status='BUSY',slot=0,
                     start_ns=102000500,end_ns=102000600,key=None,payload_sha256=None))
    with pytest.raises(AssertionError,match='Transport continued'):check(*f)


def test_unlogged_unknown_call_cannot_have_later_transport():
    f=unknown_tail();f[3].pop()
    f[3].append(dict(direction='plant-results',operation='poll',status='BUSY',slot=0,
                     start_ns=102000500,end_ns=102000600,key=None,payload_sha256=None))
    with pytest.raises(AssertionError,match='later transport'):check(*f)


def test_new_job_cannot_overwrite_unresolved_pending():
    f=fixture(('BUSY',));extend(f,11)
    identity=dict(f[0][1],activation=2,snapshot_physics=10,created_ns=120000001)
    raw=encoded(dict(identity,snapshot_payload='AA=='));f[0][2]=identity;f[1]['2']=raw
    event=dict(f[2][0],activation=2,physics=10,deadline_ns=140000000,status='PUBLISHED',payload_sha256=digest(raw))
    f[2].append(event)
    with pytest.raises(AssertionError,match='overwrote unresolved'):check(*f)
