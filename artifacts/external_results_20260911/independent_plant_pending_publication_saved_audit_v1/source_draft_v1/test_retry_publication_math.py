"""Independent synthetic ledger corruption tests; no producer helpers or arrays."""
import base64,copy,json
import numpy as np
import pytest
from retry_publication_math import check,digest,encoded


def fixture(statuses=('BUSY','PUBLISHED')):
    epoch=100000000
    identity=dict(activation=1,snapshot_physics=0,created_ns=epoch+1,
                  input_digest='a'*64,window_id='literal-window',history_digest='b'*64)
    raw=encoded(dict(identity,snapshot_payload='AA=='))
    events=[];transports=[];last=None
    for i,status in enumerate(statuses):
        guard=None if i==0 else epoch+2000000*i+100
        events.append(dict(reason='JOB_PUBLICATION',activation=1,physics=i,attempt=i+1,
            deadline_ns=epoch+20000000,retry_checked_ns=guard,input_digest=identity['input_digest'],
            source_window=identity['window_id'],payload_sha256=digest(raw),status=status))
        transports.append(dict(direction='plant-jobs',operation='publish',key=1,status=status,
            start_ns=epoch+2000000*i+200,end_ns=epoch+2000000*i+300,payload_sha256=digest(raw)))
        last=dict(identity=identity,payload=base64.b64encode(raw).decode('ascii'),deadline_ns=epoch+20000000,
                  attempts=i+1,last_attempt_index=i,last_status=status,event_recorded=True,retry_checked_ns=guard)
    summary=dict(failure=None,last_job_publication=last,pending_publication=last if statuses[-1]=='BUSY' else None,
        job_publication_attempted=len(statuses),job_publication_returned=len(statuses),job_publication_events=len(statuses),pending_expirations=0)
    arrays=dict(step_actual_start=epoch+np.arange(12)*2000000,step_actual_end=epoch+np.arange(12)*2000000+1000000)
    return [{1:identity},{'1':raw},events,transports,summary,epoch,arrays,[1] if 'PUBLISHED' in statuses else []]


def test_busy_then_success_literal_attempts():
    result=check(*fixture())
    assert result['attempted']==2 and result['retry_attempts']==1 and result['published_set_exact']
    assert not result['full_publication_coverage']


@pytest.mark.parametrize('mutation',['ordinal','physics','payload','deadline','guard','status','counter','last','pending','extra_transport','extra_success'])
def test_corrupt_attempt_ledger_rejected(mutation):
    f=fixture()
    if mutation=='ordinal':f[2][1]['attempt']=1
    elif mutation=='physics':f[2][1]['physics']=0
    elif mutation=='payload':f[3][1]['payload_sha256']='f'*64
    elif mutation=='deadline':f[2][1]['deadline_ns']+=1
    elif mutation=='guard':f[2][1]['retry_checked_ns']=120000000
    elif mutation=='status':f[2][0]['status']='FULL'
    elif mutation=='counter':f[4]['job_publication_returned']-=1
    elif mutation=='last':f[4]['last_job_publication']['identity']['history_digest']='f'*64;f[0][1]=dict(f[0][1],history_digest='b'*64)
    elif mutation=='pending':f[4]['pending_publication']=copy.deepcopy(f[4]['last_job_publication'])
    elif mutation=='extra_transport':f[3].append(copy.deepcopy(f[3][-1]))
    else:f[7].append(2)
    with pytest.raises(AssertionError):check(*f)


def test_ten_busy_attempts_then_expiry():
    f=fixture(('BUSY',)*10);last=f[4]['last_job_publication']
    f[2].append(dict(reason='PENDING_JOB_EXPIRED',activation=1,physics=10,expiration='ACTIVATION_BOUNDARY',
        observed_ns=120000000,attempts=10,last_attempt_index=9,last_status='BUSY',deadline_ns=120000000,payload_sha256=digest(f[1]['1'])))
    f[4]['pending_publication']=None;f[4]['pending_expirations']=1
    result=check(*f);assert result['expired']==1 and result['attempted']==10 and not result['pending_present']
    f[2][-1]['observed_ns']-=1
    with pytest.raises(AssertionError):check(*f)


def test_retry_after_terminal_status_rejected():
    f=fixture(('FULL','PUBLISHED'))
    with pytest.raises(AssertionError):check(*f)


def test_success_after_guard_then_scheduler_stall_is_preserved():
    f=fixture()
    f[3][1]['start_ns']=120000001;f[3][1]['end_ns']=120000100
    f[6]['step_actual_end'][1]=120000200
    result=check(*f);assert result['returned']==2
    # This is publication evidence only; unchanged protocol admission independently rejects LATE.


def test_event_overflow_retains_returned_success_without_committed_event():
    f=fixture();overflow=encoded(f[2].pop())
    f[4]['failure']={'reason':'LOGGER_CAPACITY_EXHAUSTED'}
    f[4]['job_publication_events']=1;f[4]['last_job_publication']['event_recorded']=False
    result=check(*f,event_overflow=overflow)
    assert result['returned']==2 and result['recorded_events']==1


@pytest.mark.parametrize('transport_record',['committed','overflow','absent'])
def test_exception_or_observer_failure_retains_unknown_return(transport_record):
    f=fixture();f[2].pop();last=f[4]['last_job_publication']
    last['event_recorded']=False;last['last_status']=None
    f[4]['failure']={'reason':'STEPPER_OR_CAPTURE_EXCEPTION'}
    f[4]['job_publication_returned']=1;f[4]['job_publication_events']=1;f[4]['pending_publication']=copy.deepcopy(last)
    f[7]=[];overflow=None
    if transport_record=='overflow':overflow=encoded(f[3].pop())
    elif transport_record=='absent':f[3].pop()
    result=check(*f,transport_overflow=overflow)
    assert result['attempted']==2 and result['returned']==1 and len(result['unresolved_observer_returns'])==1


def test_known_return_but_exception_before_event_is_checked():
    f=fixture();f[2].pop();f[4]['failure']={'reason':'exception while encoding event'}
    f[4]['job_publication_events']=1;f[4]['last_job_publication']['event_recorded']=False
    result=check(*f);assert result['returned']==2 and result['recorded_events']==1


def test_duplicate_tick_after_pending_expiry_rejected():
    f=fixture(('BUSY',))
    event=dict(reason='PENDING_JOB_EXPIRED',activation=1,physics=1,expiration='ORIGINAL_DEADLINE',
        observed_ns=120000000,attempts=1,last_attempt_index=0,last_status='BUSY',deadline_ns=120000000,payload_sha256=digest(f[1]['1']))
    f[2].append(event);f[4]['pending_publication']=None;f[4]['pending_expirations']=1
    f[6]['step_actual_start'][1]=120000000;f[6]['step_actual_end'][1]=120100000
    assert check(*f)['expired']==1
    f[2].append(dict(f[2][0],physics=2,attempt=2,retry_checked_ns=104000000))
    with pytest.raises(AssertionError):check(*f)


def test_boolean_cannot_replace_owned_integer_descriptor():
    f=fixture(('PUBLISHED',));f[4]['last_job_publication']['attempts']=True
    with pytest.raises(AssertionError):check(*f)


def test_expiry_clock_cannot_be_after_actual_step_finish():
    f=fixture(('BUSY',))
    f[2].append(dict(reason='PENDING_JOB_EXPIRED',activation=1,physics=1,expiration='ORIGINAL_DEADLINE',
        observed_ns=120000000,attempts=1,last_attempt_index=0,last_status='BUSY',deadline_ns=120000000,payload_sha256=digest(f[1]['1'])))
    f[4]['pending_publication']=None;f[4]['pending_expirations']=1
    with pytest.raises(AssertionError):check(*f)
