"""Four tiny synthetic counterexamples; immutable original auditor only."""
import copy,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
SOURCE=BASE.parent/'independent_plant_pending_publication_saved_audit_v1/source_draft_v1'
sys.path.insert(0,str(SOURCE))
from test_retry_publication_math import fixture
from retry_publication_math import check,digest


def expiry(f,physics):
    f[2].append(dict(reason='PENDING_JOB_EXPIRED',activation=1,physics=physics,
        expiration='ACTIVATION_BOUNDARY',observed_ns=120000000,attempts=1,last_attempt_index=0,
        last_status='BUSY',deadline_ns=120000000,payload_sha256=digest(f[1]['1'])))
    f[4]['pending_publication']=None;f[4]['pending_expirations']=1


def case(name):
    f=fixture(('BUSY',))
    f[4]['failure']={'reason':'synthetic strict failure','physics':1}
    if name=='future_expiry_beyond_terminal_tick':
        f[6]={k:v[:1].copy() for k,v in f[6].items()};f[4]['returned']=1;expiry(f,10)
    elif name=='expiry_skips_eligible_retries':
        f[6]={k:v[:11].copy() for k,v in f[6].items()};f[4]['returned']=11;expiry(f,10)
    elif name=='pending_tail_skips_completed_ticks':
        f[6]={k:v[:5].copy() for k,v in f[6].items()};f[4]['returned']=5
    elif name=='unknown_publication_before_later_native_steps':
        f=fixture();f[2].pop();f[7]=[];last=f[4]['last_job_publication']
        last.update(last_status=None,event_recorded=False)
        f[4].update(failure={'reason':'synthetic exception','physics':4},returned=4,
                    job_publication_returned=1,job_publication_events=1,pending_publication=copy.deepcopy(last))
        f[6]={k:v[:4].copy() for k,v in f[6].items()}
    return f


def main():
    names=['future_expiry_beyond_terminal_tick','expiry_skips_eligible_retries',
           'pending_tail_skips_completed_ticks','unknown_publication_before_later_native_steps']
    outcomes={}
    for name in names:
        try:check(*case(name));outcomes[name]={'incorrectly_accepted':True}
        except Exception as exc:outcomes[name]={'incorrectly_accepted':False,'error':repr(exc)}
    assert all(v['incorrectly_accepted'] for v in outcomes.values())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result=dict(author_self_review=True,actual_task_data_read=False,model_calls=0,native_steps=0,
        original_source_preparation_sha256=sha(SOURCE.parent/'source_preparation.json'),
        original_helper_sha256=sha(SOURCE/'retry_publication_math.py'),
        probe_source_sha256=sha(__file__),synthetic_counterexamples=outcomes,
        original_source_unchanged=True,actual_audit_run=False)
    with (BASE/'v1_counterexamples.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
