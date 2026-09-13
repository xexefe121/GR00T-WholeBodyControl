"""Decode saved JSON bytes only. No NumPy, framework, mailbox or native imports."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
RUN=BASE.parent/'independent_plant_clock_timeout_correction_v1'
SOURCE=RUN/'source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def subject(p):return dict(path=Path(p).as_posix(),sha256=sha(p))
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def decode(values):
    result=[]
    for index,envelope in enumerate(values):
        assert envelope['type']=='bytes'
        raw=base64.b64decode(envelope['base64'],validate=True)
        assert len(raw)==envelope['length'],index
        assert hashlib.sha256(raw).hexdigest()==envelope['sha256'],index
        result.append(json.loads(raw))
    return result
request=read(RUN/'clock_request.json');report=read(RUN/'run/report.json')
owner=read(RUN/'owner_completion.json');evidence=read(RUN/'run/evidence.json');worker=read(RUN/'run/worker.json')
assert sha(RUN/'clock_request.json')==report['request_sha256']==owner['request_sha256']
assert sha(RUN/'run/report.json')==owner['report_sha256']=='b63896a22b00a032f9d1ac5e9fcb420d796778607374d11e4a1ad7fec3ec52fc'
assert sha(RUN/'run/evidence.json')==report['schema']['evidence_sha256']=='11e250e56015cc22290d7f633153b3af6fb08f30efb672f45baf79a0308d7791'
for name in ('report.json','evidence.json','worker.json'):
    p=RUN/'run'/name;assert owner['output_hashes'][p.as_posix()]==sha(p)
assert owner['accounting_passed'] and owner['process_absence_proven']
assert owner['diagnostic_passed'] is False and report['component_preliminary_pass'] is False
input_pins={p['path']:p['sha256'] for p in request['input_files']}
source_pins={}
for name in ('shared_mailbox','session','clock_core','dummy_worker','recorded_protocol','evidence','packet_io'):
    path=SOURCE/(name+'.py');digest=sha(path)
    assert input_pins[path.as_posix()]==digest
    assert report['imported_sources'][name]['sha256']==digest
    source_pins[name+'.py']=subject(path)
run_source=SOURCE/'run_clock.py';assert input_pins[run_source.as_posix()]==sha(run_source)
source_pins['run_clock.py']=subject(run_source)
plant=decode(evidence['transport_records']);foundation=decode(evidence['foundation_events']);outer=decode(evidence['outer_cycles'])
assert len(plant)==839 and len(foundation)==889 and len(outer)==4694
wp=worker['events']
published={x['key']:x for x in plant if x['direction']=='plant-jobs' and x['status']=='PUBLISHED'}
busy=[x for x in plant if x['status']=='BUSY']
taken={x['key']:x for x in wp if x.get('direction')=='worker-jobs' and x['status']=='TAKEN'}
replies={x['activation']:x for x in wp if x.get('reason')=='RECORDED_REPLY'}
returned={x['key']:x for x in wp if x.get('direction')=='worker-results'}
received={x['key']:x for x in plant if x['direction']=='plant-results'}
sealed={x['activation']:x for x in foundation if x['reason']=='RESULT_SEALED'}
issued={x['activation']:x for x in foundation if x['reason']=='JOB_PUBLICATION'}
assert len(busy)==1 and busy[0]['direction']=='plant-jobs' and busy[0]['key']==420
assert all(set(v)==set(range(1,420)) for v in (published,taken,replies,returned,received,sealed))
assert set(issued)==set(range(1,421))
epoch=report['epoch_ns'];assert epoch==124611372988
for activation in range(1,420):
    p,t,r,out,got,s=published[activation],taken[activation],replies[activation],returned[activation],received[activation],sealed[activation]
    assert p['payload_sha256']==t['payload_sha256']==r['job_sha256']
    payload=base64.b64decode(s['payload'],validate=True);result=json.loads(payload)
    assert hashlib.sha256(payload).hexdigest()==r['result_sha256']==out['payload_sha256']==got['payload_sha256']
    assert out['status']=='PUBLISHED' and got['status']=='TAKEN' and r['status']=='PUBLISHED'
    assert result['identity']['activation']==activation and result['identity']['input_digest']==issued[activation]['input_digest']
    assert result['identity']['snapshot_control']==activation-1 and result['identity']['snapshot_physics']==(activation-1)*10
    assert result['identity']['binding']['run']==request['run_id'] and result['identity']['binding']['epoch']==request['input_epoch']
    assert result['identity']['binding']['reference_hash']==input_pins[request['roles']['reference']]
    assert result['identity']['binding']['model_hash']==request['expected_model_sha256']
    assert result['identity']['window_id'].endswith(request['command_table_sha256'])
    assert result['identity']['created_ns']<=result['completed_ns']==r['completed_ns']<=s['received_ns']<epoch+activation*20000000
    assert t['slot']==got['slot']==activation%2
    assert t['version']==got['version']==(activation+1)//2
failed=busy[0];deadline=epoch+420*20000000
assert (failed['start_ns'],failed['end_ns'],deadline)==(132991903623,132991908848,133011372988)
assert issued[420]['status']=='BUSY' and issued[420]['physics']==4190
held=[x for x in foundation if x['reason']=='HELD_COMMAND_INTERVAL']
assert [x['control'] for x in held]==list(range(420,470))
assert all(x['command_id']==evidence['control_command_ids'][419] for x in held)
assert all(x==evidence['control_command_ids'][419] for x in evidence['control_command_ids'][420:])
assert report['session']['foundation']['input_fault']==dict(reason='COMMAND_DEADLINE_MISSED',control=420,nominal_deadline=deadline)
assert worker['failure'] is None and worker['overflow'] is None and worker['polls']==8543
wb=[x for x in wp if x.get('direction')=='worker-jobs' and x['status']=='BUSY']
overlaps=[]
for x in wb:
    matching=[a for a,p in published.items() if a%2==x['slot'] and p['start_ns']<=x['end_ns'] and x['start_ns']<=p['end_ns']]
    assert len(matching)==1
    overlaps.append(dict(worker_poll=x,matching_slot_plant_publication=published[matching[0]]))
assert len(overlaps)==10
assert all(x['index']==i for i,x in enumerate(outer))
before_expiry=outer[4190:4200]
assert all(x['cycle_return_ns']<deadline for x in before_expiry)
summary=report['session']['foundation']
assert summary['attempted']==summary['returned']==summary['captured']==4694 and summary['verified']==4693
assert summary['first_deadline_failure']['index']==610 and summary['deadline_misses']==6
timeline=[]
for label,rows in [('plant',plant),('worker',wp),('foundation',foundation)]:
    for x in rows:
        if x.get('key',x.get('activation',x.get('control',-1))) in (418,419,420,421):
            timeline.append(dict(origin=label,**{k:v for k,v in x.items() if k!='payload'}))
write(BASE/'decoded_timeline.json',dict(records=timeline,worker_BUSY_overlap_examples=overlaps,
    actual_outer_cycles_before_420_deadline=before_expiry))
subjects={name:subject(RUN/name) for name in ('clock_request.json','owner_completion.json','run/report.json','run/evidence.json','run/worker.json')}
subjects['diagnostic_source']=subject(__file__);subjects['timeline']=subject(BASE/'decoded_timeline.json')
result=dict(saved_byte_diagnosis_passed=True,subjects=subjects,source_subjects=source_pins,
    verified_byte_envelopes=dict(transport=839,foundation=889,outer_cycles=4694),
    successfully_hash_correlated_jobs_results=419,
    publication_failure=dict(activation=420,slot=0,status='BUSY',call_start_ns=failed['start_ns'],call_end_ns=failed['end_ns'],
        call_duration_ns=failed['end_ns']-failed['start_ns'],original_deadline_ns=deadline,
        remaining_ns_after_failed_attempt=deadline-failed['end_ns'],attempt_count=1,
        worker_received=False,worker_completed=False,payload_sha256=failed['payload_sha256'],
        snapshot_input_digest=issued[420]['input_digest']),
    causal_chain=['Single nonblocking publication returned BUSY at physics4190; PUBLISHED membership was never granted.',
        'No deferred retry exists in boundary/tick; worker never took job420 and no result420 was sealed.',
        'Control420 latched COMMAND_DEADLINE_MISSED; later job creation stopped and command419 was held for controls420..469.',
        'Native joint-bound failure later ended at4694 attempted/returned/captured,4693verified; no replay or causal physics counterfactual performed.'],
    lock_evidence=dict(worker_polls=8543,worker_slot_attempts=17086,worker_TAKEN=419,worker_BUSY=10,
        unlogged_EMPTY_slot_returns=16657,all_10_worker_BUSY_intervals_overlap_matching_plant_publication=True,
        actual_420_worker_lock_acquire_release_interval_saved=False,
        inference='BUSY proves failed immediate lock acquisition. With the recorded two-endpoint topology, concurrent worker polling is the expected lock holder; its exact EMPTY-poll interval and lock duration are unrecorded.',
        failed_call_duration_is_not_lock_hold_duration=True),
    separate_existing_timing_failure=dict(first=summary['first_deadline_failure'],deadline_misses=6,
        outer_deadline_miss_indices=report['session']['outer_deadline_miss_indices'],
        BUSY_retry_would_not_qualify_this_existing_run=True),
    proposed_minimal_change=dict(source_only=True,mutable_pending_slots=1,
        immutable_job_and_serialized_payload=True,retry_statuses=['BUSY'],
        max_total_attempts_per_job=10,max_attempts_per_physics_tick=1,
        retry_tick_rule='Original attempt at predecessor boundary; at most one deferred attempt on each of the following nine physics ticks, only before the original activation boundary and deadline.',
        publication_admission='Mark published only on actual PUBLISHED; once successful never publish again. Success returning after deadline remains recorded but cannot cause late result admission.',
        expiry='No retry at/after activation boundary, after original deadline, or after input/plant fault. Preserve pending bytes/digest/attempt ledger and original boundary deadline miss.',
        unchanged=['epoch/deadlines','1819commands/18190native-step budget','fourMJB budget','history ownership','job snapshot/created time/identity','worker poll/watchdog budgets','fixed ledger capacities','strict result admission/native gates'],
        forbidden=['blocking acquire','immediate spin/retry loop','sleep/yield to await publication','recreated job from later state/history','deadline rebase','native skip or replay','FULL overwrite','second worker computation'],
        separate_unobserved_risk='Worker result publication remains single-attempt; no result-side BUSY occurred in this run.'),
    no_actual_clock_or_mailbox_execution=True,model_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
    existing_source_mutation=False,behavioral_qualification=False)
write(BASE/'report.json',result)
print(json.dumps(dict(report=subject(BASE/'report.json'),verified_envelopes=6422,remaining_ns=deadline-failed['end_ns'])))
