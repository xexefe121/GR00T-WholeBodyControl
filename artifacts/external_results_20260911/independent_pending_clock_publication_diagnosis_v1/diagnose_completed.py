"""Completed saved timestamp/transport diagnosis; zero new model/native calls."""
import base64,hashlib,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
RUN=NEW/'independent_plant_pending_publication_v1'
AUDIT=NEW/'independent_plant_pending_publication_saved_actual_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def normalized(path):
    s=str(path).replace(chr(92),'/')
    if s.startswith('/mnt/') and s[6:7]=='/':s=s[5]+':'+s[6:]
    return s.casefold()
def decode(rows):
    result=[]
    for row in rows:
        assert row['type']=='bytes'
        raw=base64.b64decode(row['base64'],validate=True)
        assert len(raw)==row['length'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        result.append(json.loads(raw))
    return result
def stats(a):return dict(zip(('p50','p95','p99','max'),map(float,np.percentile(a,[50,95,99,100]))))

audit_path=AUDIT/'results_v1/report.json';audit=read(audit_path)
assert sha(audit_path)=='6d53378fdb7b439487728d5b8a56796ee6343f068dc42a0f8558b3613e3e0d24'
assert audit['evidence_integrity_passed'] is True and audit['component_qualified'] is False
owner_path=AUDIT/'owner_completion_dispatch_v3.json';owner=read(owner_path)
assert owner['completion_accounting_passed'] is True and owner['processes_absent'] is True
assert owner['report_sha256']==sha(audit_path) and owner['evidence_integrity_passed'] is True
pins={normalized(k):v for k,v in audit['input_sha256'].items()}
inputs={audit_path.as_posix():sha(audit_path),owner_path.as_posix():sha(owner_path)}
def bind(p):
    p=Path(p);actual=sha(p);assert pins[normalized(p)]==actual
    inputs[p.as_posix()]=actual;return p
request=read(bind(RUN/'clock_request.json'));report=read(bind(RUN/'run/report.json'))
evidence=read(bind(RUN/'run/evidence.json'));worker=read(bind(RUN/'run/worker.json'))
trace_path=bind(RUN/'run/trace.npz')
source_paths=[bind(RUN/'source_draft_v1'/n) for n in ('clock_core.py','session.py','dummy_worker.py','run_clock.py','shared_mailbox.py','recorded_protocol.py')]
plant=decode(evidence['transport_records']);foundation=decode(evidence['foundation_events']);outer=decode(evidence['outer_cycles'])
epoch=report['epoch_ns'];D=lambda a:epoch+a*20000000
jobs=[r for r in plant if r['direction']=='plant-jobs']
busy_keys=sorted({r['key'] for r in jobs if r['status']=='BUSY'})
assert busy_keys==[136,192,213,614]
recovered=[]
for key in busy_keys:
    attempts=[r for r in jobs if r['key']==key]
    events=[r for r in foundation if r['reason']=='JOB_PUBLICATION' and r['activation']==key]
    sealed=[r for r in foundation if r['reason']=='RESULT_SEALED' and r['activation']==key]
    assert len(attempts)==len(events)==2 and [r['status'] for r in attempts]==['BUSY','PUBLISHED']
    assert attempts[0]['payload_sha256']==attempts[1]['payload_sha256']
    assert [r['attempt'] for r in events]==[1,2] and [r['physics'] for r in events]==[(key-1)*10,(key-1)*10+1]
    assert events[0]['input_digest']==events[1]['input_digest']
    assert len(sealed)==1 and sealed[0]['received_ns']<D(key)
    recovered.append(dict(activation=key,attempts=attempts,publication_events=events,original_deadline_ns=D(key),
        result_admitted_ns=sealed[0]['received_ns'],remaining_ns_at_admission=D(key)-sealed[0]['received_ns']))
key=656
pub=[r for r in jobs if r['key']==key]
taken=[r for r in worker['events'] if r.get('direction')=='worker-jobs' and r.get('key')==key]
out=[r for r in worker['events'] if r.get('direction')=='worker-results' and r.get('key')==key]
reply=[r for r in worker['events'] if r.get('reason')=='RECORDED_REPLY' and r['activation']==key]
assert len(pub)==len(taken)==len(out)==len(reply)==1
pub,taken,out,reply=pub[0],taken[0],out[0],reply[0]
assert pub['status']=='PUBLISHED' and taken['status']=='TAKEN' and out['status']==reply['status']=='BUSY'
assert pub['payload_sha256']==taken['payload_sha256']==reply['job_sha256']
assert out['payload_sha256']==reply['result_sha256']
assert not any(r.get('direction')=='plant-results' and r.get('key')==key for r in plant)
assert not any(r['reason']=='RESULT_SEALED' and r['activation']==key for r in foundation)
held=[r['control'] for r in foundation if r['reason']=='HELD_COMMAND_INTERVAL'];assert held==list(range(656,683))
failure=dict(activation=656,deadline_ns=D(656),job_published=pub,worker_taken=taken,worker_reply=reply,result_publication=out,
    result_publish_call_ns=out['end_ns']-out['start_ns'],remaining_ns_after_BUSY=D(656)-out['end_ns'],
    job_publish_return_to_worker_take_end_ns=taken['end_ns']-pub['end_ns'],
    received_to_stamped_completion_ns=reply['completed_ns']-reply['received_ns'],
    stamped_completion_to_publish_start_ns=out['start_ns']-reply['completed_ns'],
    worker_published_status_return_ns=reply['published_return_ns'],held_controls=held,
    result_received_by_plant=False,result_admitted=False,
    completion_stamp_precedes_reply_encoding_per_source=True,lock_hold_duration_not_observed=True,
    interpretation='Worker computed one recorded result but its sole nonblocking publication returned BUSY; no pending-result retry exists in this source.')
assert failure['remaining_ns_after_BUSY']==17413713
with np.load(trace_path,allow_pickle=False) as a:
    ns,ne,starts,ends=[a[k].copy() for k in ('step_nominal_start','step_nominal_end','step_actual_start','step_actual_end')]
assert all(v.shape==(6824,) for v in (ns,ne,starts,ends))
assert np.array_equal(ns,epoch+np.arange(6824,dtype=np.int64)*2000000) and np.array_equal(ne,ns+2000000)
assert np.all(starts>=ns) and np.all(ends>=starts)
body=ends-starts;wake=starts-ns;late=np.maximum(ends-ne,0)
indices=np.flatnonzero(late).tolist();assert indices==audit['timing']['captured_step_deadline_misses']
outer_indices=[r['index'] for r in outer if r['cycle_return_ns']>r['fixed_nominal_end_ns']]
assert outer_indices==audit['timing']['outer_deadline_misses']
outer_only=sorted(set(outer_indices)-set(indices));assert outer_only==[5314]
misses=[dict(index=i,wake_delay_ns=int(wake[i]),body_ns=int(body[i]),overrun_ns=int(late[i]),wake_already_past_end=bool(starts[i]>ne[i])) for i in indices]
outer_gap=[dict(index=i,native_capture_end_ns=int(ends[i]),nominal_end_ns=int(ne[i]),outer_return_ns=outer[i]['cycle_return_ns'],
    outer_accounting_gap_ns=outer[i]['cycle_return_ns']-int(ends[i]),native_step_missed=bool(late[i])) for i in outer_only]
stage_paths=sorted((RUN/'stage_receipts').glob('*.json'));stages=[read(bind(p)) for p in stage_paths]
assert len(stages)==6 and [r['stage'] for r in stages]==['setup_started','native_setup_started','native_restored','epoch_armed','plant_or_setup_exit','preservation_complete']
armed,finished=stages[3],stages[4]
assert armed['epoch_ns']==finished['epoch_ns']==epoch and armed['epoch_chosen_ns']==finished['epoch_chosen_ns']
assert armed['epoch_chosen_ns']<armed['allocation_finished_ns']<finished['setup_finished_ns']<epoch
assert stages[-1]['watchdog']['timeouts']==[]
stage_summary=dict(setup_started_ns=stages[0]['record_created_ns'],native_setup_started_ns=stages[1]['record_created_ns'],
    native_restored_ns=stages[2]['record_created_ns'],epoch_chosen_ns=armed['epoch_chosen_ns'],epoch_ns=epoch,
    allocation_finished_ns=armed['allocation_finished_ns'],setup_finished_ns=finished['setup_finished_ns'],
    plant_exit_ns=finished['record_created_ns'],preservation_finished_ns=stages[-1]['record_created_ns'],
    watchdog_budgets=stages[-1]['watchdog']['budgets'],watchdog_timeouts=[],deadline_rebased=False)
for p in (Path(__file__),BASE/'inspect_saved.py',BASE/'inspection.json'):
    inputs[p.as_posix()]=sha(p)
result=dict(saved_diagnosis_passed=True,audit_evidence_integrity_passed=True,audit_comparisons=audit['comparisons'],
    actual_native_steps_observed=6824,new_native_steps=0,model_calls=0,optimizer_updates=0,
    recovered_job_BUSY=recovered,worker_result_failure=failure,
    separate_timing=dict(captured_step_misses=len(indices),outer_misses=len(outer_indices),misses=misses,outer_only_details=outer_gap,
        first_miss=misses[0],wake_ns=stats(wake),body_ns=stats(body),
        returned_failed_final_step_was_captured=True,extra_outer_miss_is_completed_index5314=True),
    stages=stage_summary,input_sha256=inputs,
    next_source_only_change='Retain the same completed result/job/deadline on unambiguous worker-result BUSY; bounded deferred attempts, no recompute or new-job consumption while pending, unchanged plant admission.',
    limitations=['No new clock/physics counterfactual; recovered publications do not establish timing or balance qualification.',
        'Transport BUSY proves failed immediate lock acquisition; unlogged EMPTY polling prevents exact lock-owner interval/duration attribution.',
        'Step body includes Python/native/checking/OS preemption. No GC event or pause attribution was recorded.',
        'Fifteen native-loop deadline misses remain independent of the worker-result publication loss.'],
    behavioral_qualification=False,actual_next_run_selected=False)
write(BASE/'report.json',result)
print(json.dumps(dict(report_sha256=sha(BASE/'report.json'),recovered_BUSY_keys=busy_keys,result656_remaining_ns=failure['remaining_ns_after_BUSY'],
    native_step_misses=len(indices),outer_misses=len(outer_indices),outer_only=outer_only)))
