"""Saved timestamp/transport diagnosis. No new physics, workers or model calls."""
from pathlib import Path
import base64,collections,hashlib,json
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
RUN=NEW/'independent_plant_pending_result_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def normalized(p):return str(Path(p).resolve()).replace(chr(92),'/').casefold()
def write(p,j):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False);f.write('\n')
def decode(rows):
    decoded=[]
    for row in rows:
        assert row['type']=='bytes'
        raw=base64.b64decode(row['base64'],validate=True)
        assert len(raw)==row['length'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        decoded.append(json.loads(raw))
    return decoded
def compact(row):return {k:v for k,v in row.items() if k not in ('payload','base64','state','history')}
def quantiles(a):return dict(zip(('p50','p95','p99','max'),map(float,np.percentile(a,[50,95,99,100]))))

def main():
    owner_path=RUN/'owner_completion.json';owner=read(owner_path)
    assert sha(owner_path)=='968a2e8bb369a9168c0383503cb4775c1bcf87bf5482703d8c233ca3476017ae'
    assert owner['accounting_passed'] is True and owner['process_absence_proven'] is True
    assert owner['diagnostic_passed'] is False and owner['raw_python_exit_code']==2
    pins={}
    for mapping in (owner['input_hashes'],owner['output_hashes']):
        for p,h in mapping.items():
            key=normalized(p);assert key not in pins or pins[key]==h;pins[key]=h
    inputs={owner_path.as_posix():sha(owner_path)}
    def bind(p):
        p=Path(p).resolve();h=sha(p);assert pins[normalized(p)]==h
        inputs[p.as_posix()]=h;return p
    request=read(bind(RUN/'clock_request.json'));report_path=bind(RUN/'run/report.json');report=read(report_path)
    assert sha(report_path)==owner['report_sha256']=='209b04cc616fce8b51ba76d7af593533f1099427da33c4b30fa90db0b8da3bff'
    e=read(bind(RUN/'run/evidence.json'));worker=read(bind(RUN/'run/worker.json'))
    trace_path=bind(RUN/'run/trace.npz')
    for name in ('clock_core.py','run_clock.py','session.py','pending_result.py','dummy_worker.py','recorded_protocol.py','shared_mailbox.py'):
        bind(RUN/'source_draft_v1'/name)
    stages=[read(bind(p)) for p in sorted((RUN/'stage_receipts').glob('*.json'))]
    assert len(stages)==6
    transport=decode(e['transport_records']);events=decode(e['foundation_events']);outer=decode(e['outer_cycles'])
    f=report['session']['foundation'];epoch=report['epoch_ns'];deadline=lambda c:epoch+c*20000000
    assert f['input_fault']['reason']=='COMMAND_DEADLINE_MISSED';key=f['input_fault']['control'];assert key==1211
    jobrows=[r for r in transport if r.get('direction')=='plant-jobs'];target=[r for r in jobrows if r['key']==key]
    taken=[r for r in worker['events'] if r.get('direction')=='worker-jobs' and r.get('key')==key]
    terminals=[r for r in worker['events'] if r.get('reason')=='WORKER_RESULT_TERMINAL' and r['activation']==key]
    assert len(target)==len(taken)==len(terminals)==1
    pub,take,terminal=target[0],taken[0],terminals[0]
    retry=worker['result_retry'];pending=retry['pending'];identity=pending['identity']
    assert pub['payload_sha256']==take['payload_sha256']==terminal['job_sha256']==pending['source']['payload_sha256']
    job=json.loads(base64.b64decode(pending['source']['payload'],validate=True))
    assert job['deadline_ns']==identity['deadline_ns']==terminal['deadline_ns']==deadline(key)
    assert identity['created_ns']>deadline(key)
    assert pub['status']=='PUBLISHED' and take['status']=='TAKEN'
    assert terminal['termination']=='EXPIRED_BEFORE_REPLY' and terminal['attempts']==0 and terminal['result_sha256'] is None
    assert pending['completed_ns'] is None and pending['payload'] is None and pending['attempts']==0
    result_pubs=[r for r in worker['events'] if r.get('reason')=='WORKER_RESULT_PUBLICATION']
    assert len(result_pubs)==1210 and all(r['status']=='PUBLISHED' for r in result_pubs)
    assert retry['counts']['publications_BUSY']==0 and retry['counts']['replies_attempted']==retry['counts']['replies_completed']==1210
    assert not any(r.get('direction')=='plant-results' and r.get('key')==key for r in transport)
    assert not any(r.get('reason')=='RESULT_SEALED' and r.get('activation')==key for r in events)
    busy_recoveries=[]
    for c in sorted({r['key'] for r in jobrows if r['status']=='BUSY'}):
        rows=[r for r in jobrows if r['key']==c]
        assert len({r['payload_sha256'] for r in rows})==1 and all(r['status']=='BUSY' for r in rows[:-1]) and rows[-1]['status']=='PUBLISHED'
        admitted=[r for r in events if r.get('reason')=='RESULT_SEALED' and r.get('activation')==c]
        assert len(admitted)==1 and admitted[0]['received_ns']<=deadline(c)
        busy_recoveries.append(dict(control=c,attempts=[compact(r) for r in rows],deadline_ns=deadline(c),admitted_ns=admitted[0]['received_ns']))
    with np.load(trace_path,allow_pickle=False) as arrays:
        ns,ne,starts,ends=[arrays[k].copy() for k in ('step_nominal_start','step_nominal_end','step_actual_start','step_actual_end')]
    n=report['api_counters']['step_returned'];assert n==12663 and all(a.shape==(n,) for a in (ns,ne,starts,ends))
    assert np.array_equal(ns,epoch+np.arange(n,dtype=np.int64)*2000000) and np.array_equal(ne,ns+2000000)
    assert np.all(starts>=ns) and np.all(ends>=starts)
    body=ends-starts;wake=starts-ns;late=np.maximum(ends-ne,0)
    missed=np.flatnonzero(late).tolist();outer_missed=[r['index'] for r in outer if r['cycle_return_ns']>r['fixed_nominal_end_ns']]
    assert len(missed)==f['deadline_misses']==40 and outer_missed==report['session']['outer_deadline_miss_indices']
    assert missed==outer_missed
    def timing(i):return dict(index=i,nominal_start_ns=int(ns[i]),nominal_end_ns=int(ne[i]),actual_start_ns=int(starts[i]),
        actual_end_ns=int(ends[i]),wake_delay_ns=int(wake[i]),body_ns=int(body[i]),overrun_ns=int(late[i]),outer_return_ns=outer[i]['cycle_return_ns'])
    preceding=(key-1)*10-1;assert preceding==12099
    held=report['session']['commands']['held_controls'];assert held==list(range(key,1267))
    assert held==report['session']['commands']['mismatch_controls']
    assert [r['control'] for r in events if r.get('reason')=='HELD_COMMAND_INTERVAL']==held
    assert report['epoch_rebased'] is False and stages[-1]['watchdog']['timeouts']==[]
    assert stages[3]['epoch_ns']==stages[4]['epoch_ns']==epoch
    assert stages[3]['epoch_chosen_ns']<stages[3]['allocation_finished_ns']<stages[4]['setup_finished_ns']<epoch
    inputs[Path(__file__).resolve().as_posix()]=sha(Path(__file__))
    result=dict(saved_diagnosis_passed=True,independent_saved_audit_pending=True,owner_accounting_passed=True,
        report_sha256=sha(report_path),owner_sha256=sha(owner_path),actual_native_steps_observed=n,
        new_native_steps=0,model_calls=0,optimizer_updates=0,worker_processes_started=0,
        failure=dict(control=key,deadline_ns=deadline(key),job_identity=identity,job_publication=compact(pub),worker_take=compact(take),worker_terminal=terminal,
            created_after_deadline_ns=identity['created_ns']-deadline(key),published_after_deadline_ns=pub['end_ns']-deadline(key),
            taken_after_deadline_ns=take['end_ns']-deadline(key),terminal_after_deadline_ns=terminal['observed_ns']-deadline(key),
            publication_return_to_take_end_ns=take['end_ns']-pub['end_ns'],preceding_tick=timing(preceding),
            reply_attempts=0,result_publication_attempts=0,held_controls=held,held_count=len(held)),
        worker_counts=retry['counts'],job_publication_status_counts=dict(collections.Counter(r['status'] for r in jobrows)),
        recovered_job_BUSY=busy_recoveries,worker_result_retry_exercised=False,
        timing=dict(first_miss=timing(missed[0]),all_misses=[timing(i) for i in missed],
            captured_step_misses=len(missed),outer_misses=len(outer_missed),body_ns=quantiles(body),wake_ns=quantiles(wake),
            around_expiry=[timing(i) for i in range(12097,12116)],max_debt=f['max_debt']),
        strict_failure=f['failure'],stage_times=[dict(stage=r['stage'],record_created_ns=r['record_created_ns']) for r in stages],
        unchanged_watchdog_budgets=stages[-1]['watchdog']['budgets'],watchdog_timeouts=[],all_four_MJB_exact=report['all_four_MJB_exact'],
        input_sha256=inputs,behavioral_qualification=False,actual_next_run_selected=False,
        interpretation='The plant constructed and published job1211 after its original deadline following a delayed preceding tick; the worker correctly rejected the already-expired job before computing a reply. No worker-result BUSY retry occurred in this trial.',
        limitations=['Independent full saved audit has not run yet; this report checks named saved observations and preserves qualification failure.',
            'Step body includes native execution, checks, Python and scheduling/preemption; no GC callback or profiler data identifies the source of delay.',
            'No counterfactual timed run; recovered plant BUSY publications and zero worker-result BUSY do not prove retry effectiveness or timing/balance qualification.',
            'EMPTY worker polls are not logged; exact lock ownership and duration cannot be reconstructed.'])
    for p,h in inputs.items():assert sha(p)==h,'Input changed during saved diagnosis'
    write(BASE/'report.json',result)
    print(json.dumps(dict(report_sha256=sha(BASE/'report.json'),first_miss=missed[0],preceding_tick_body_ns=int(body[preceding]),
        late_job_ns=identity['created_ns']-deadline(key),recovered_job_BUSY=[v['control'] for v in busy_recoveries],worker_result_BUSY=0)))

if __name__=='__main__':main()
