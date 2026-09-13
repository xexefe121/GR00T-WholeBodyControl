"""Hand-built synthetic saved envelopes; no producer or task runtime imports."""
import base64,copy,json,unittest
from worker_result_math import check,encoded,digest,COUNTERS,CONTRACT


def b64(v):return base64.b64encode(v).decode()


def one_case(statuses=('BUSY','PUBLISHED'),deadline=120000000):
    epoch=100000000
    identity=dict(binding={'run':'synthetic','epoch':3},sequence=0,snapshot_control=0,snapshot_physics=0,
        activation=1,window_id='synthetic:1',input_digest='a'*64,history_digest='b'*64,schedule_digest='c'*64,
        created_ns=epoch+1,deadline_ns=deadline)
    job=encoded(dict(identity,snapshot_payload=b64(b'synthetic-owned-input')))
    command={'command_id':'saved:1','origin':'synthetic','target':b64(bytes(184)),'raw_action':b64(bytes(92))}
    command['raw_action']=b64(bytes(88)+b'\x00\x00\x00\x80')  # Preserve signed zero.
    ready=dict(reason='WORKER_RESULT_READY',activation=1,identity=identity,received_ns=epoch+30,
        completed_ns=epoch+100,payload_ready_ns=epoch+110,job_sha256=digest(job))
    result=encoded(dict(identity=identity,command=command,completed_ns=ready['completed_ns']))
    ready.update(payload=b64(result),result_sha256=digest(result))
    events=[dict(direction='worker-jobs',operation='poll',slot=1,status='TAKEN',key=1,version=1,
        payload_sha256=digest(job),start_ns=epoch+10,end_ns=epoch+20),ready]
    for i,status in enumerate(statuses):
        start=epoch+200+i*1000000;end=start+100
        events.append(dict(direction='worker-results',operation='publish',key=1,status=status,
            payload_sha256=digest(result),start_ns=start+10,end_ns=end-10))
        events.append(dict(reason='WORKER_RESULT_PUBLICATION',activation=1,iteration=i,attempt=i+1,
            start_ns=start,end_ns=end,status=status,deadline_ns=deadline,job_sha256=digest(job),result_sha256=digest(result)))
    last_status=statuses[-1] if statuses else None;attempts=len(statuses)
    if last_status=='PUBLISHED':reason='PUBLISHED';terminal_ns=end
    elif attempts==20:reason='ATTEMPT_LIMIT';terminal_ns=end
    elif last_status=='BUSY':reason='ORIGINAL_DEADLINE';terminal_ns=deadline
    elif last_status is None:reason='ORIGINAL_DEADLINE';terminal_ns=deadline
    else:reason='PUBLICATION_'+last_status;terminal_ns=end
    events.append(dict(reason='WORKER_RESULT_TERMINAL',activation=1,termination=reason,observed_ns=terminal_ns,
        deadline_ns=deadline,attempts=attempts,last_status=last_status,call_returned=bool(attempts),
        job_sha256=digest(job),result_sha256=digest(result)))
    extra=reason=='ORIGINAL_DEADLINE';failed=reason!='PUBLISHED'
    iterations=attempts+int(extra)
    counts={k:0 for k in COUNTERS}
    counts.update(iterations_started=iterations,iterations_completed=iterations-int(failed),polls_attempted=1,polls_returned=1,
        jobs_taken=1,jobs_decoded=1,replies_attempted=1,replies_completed=1,ready_events=1,
        publications_attempted=attempts,publications_returned=attempts,publications_published=statuses.count('PUBLISHED'),
        publications_BUSY=statuses.count('BUSY'),publication_events=attempts,terminals=1,terminal_events=1)
    descriptor=dict(source=dict(slot=1,key=1,version=1,payload=b64(job),payload_sha256=digest(job)),received_ns=ready['received_ns'],
        identity=identity,completed_ns=ready['completed_ns'],payload=b64(result),payload_sha256=digest(result),
        payload_ready_ns=ready['payload_ready_ns'],ready_event_recorded=True,attempts=attempts,
        last_iteration=attempts-1 if attempts else None,attempt_start_ns=start if attempts else None,
        attempt_end_ns=end if attempts else None,last_status=last_status,call_returned=bool(attempts),
        returned_status_evidence=encoded({'type':'builtins.str','value':last_status}).decode() if attempts else None,
        attempt_event_recorded=bool(attempts),terminal_reason=reason,terminal_ns=terminal_ns,terminal_event_recorded=True)
    failure={'type':'WorkerResultFailure','detail':reason} if failed else None
    state=dict(contract=CONTRACT,max_attempts=20,counts=counts,phase='RESULT_PUBLICATION',closed=True,
        pending=copy.deepcopy(descriptor) if failed else None,last_result=descriptor,buffered_jobs=[],
        last_poll_return=None,clock_failure=None,failure=failure)
    worker=dict(recorded_command_table_sha256='d'*64,native_steps=0,model_calls=0,pid=3,polls=counts['iterations_completed'],
        start_ns=epoch-1000,end_ns=terminal_ns+1000,failure=failure,overflow=None,events=events,result_retry=state)
    return worker,{1:identity},{'1':job},[None,command],epoch,'d'*64,[digest(result)] if last_status=='PUBLISHED' else []


def sync_last(worker):
    if worker['result_retry']['pending'] is not None:
        worker['result_retry']['pending']=copy.deepcopy(worker['result_retry']['last_result'])


class WorkerMathTests(unittest.TestCase):
    def test_successful_retry_is_same_result_with_separate_counts(self):
        r=check(*one_case());self.assertEqual(r['publication_attempts'],2);self.assertEqual(r['retry_attempts'],1)
        self.assertEqual(r['published_publications'],1);self.assertTrue(r['counters_exact']);self.assertFalse(r['full_worker_coverage'])
    def test_original_deadline_expiry_after_busy_preserved(self):
        r=check(*one_case(('BUSY',)));self.assertTrue(r['worker_failed']);self.assertEqual(r['published_publications'],0)
    def test_twenty_busy_attempts_exact_cap(self):
        # Last attempt is at119000200, still before120000000.
        r=check(*one_case(tuple(['BUSY']*20)));self.assertEqual(r['publication_attempts'],20)
    def test_retry_changed_payload_or_deadline_rejected(self):
        for field,value in [('result_sha256','0'*64),('deadline_ns',120000001),('attempt',4),('iteration',0)]:
            a=list(one_case());a[0]['events'][-2][field]=value
            with self.subTest(field=field),self.assertRaises(AssertionError):check(*a)
    def test_publication_at_deadline_rejected(self):
        a=list(one_case());a[0]['events'][-2]['start_ns']=120000000
        with self.assertRaises(AssertionError):check(*a)
    def test_busy_expiry_early_rejected(self):
        a=list(one_case(('BUSY',)));a[0]['events'][-1]['observed_ns']=119999999
        with self.assertRaisesRegex(AssertionError,'Expiry'):check(*a)
    def test_more_than_twenty_attempts_rejected(self):
        with self.assertRaises(AssertionError):check(*one_case(tuple(['BUSY']*20+['PUBLISHED'])))
    def test_retry_after_full_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Only BUSY'):check(*one_case(('FULL','PUBLISHED')))
    def test_new_job_poll_while_pending_rejected(self):
        a=list(one_case());w=a[0]
        w['events'].insert(4,dict(direction='worker-jobs',operation='poll',slot=0,status='BUSY',key=None,version=None,payload_sha256=None,start_ns=100001000,end_ns=100001010))
        with self.assertRaisesRegex(AssertionError,'polling while'):check(*a)
    def test_duplicate_ready_and_terminal_rejected(self):
        for index in (1,-1):
            a=list(one_case());a[0]['events'].append(copy.deepcopy(a[0]['events'][index]))
            with self.subTest(index=index),self.assertRaises(AssertionError):check(*a)
    def test_missing_known_attempt_rejected(self):
        a=list(one_case());del a[0]['events'][-2]
        with self.assertRaises(AssertionError):check(*a)
    def test_each_reported_counter_is_checked(self):
        for field in COUNTERS:
            a=list(one_case());a[0]['result_retry']['counts'][field]+=1
            with self.subTest(field=field),self.assertRaises(AssertionError):check(*a)
    def test_wrong_job_deadline_rejected(self):
        a=list(one_case());a[1][1]['deadline_ns']=125000000
        with self.assertRaisesRegex(AssertionError,'deadline'):check(*a)
    def test_missing_result_retry_schema_rejected(self):
        a=list(one_case());del a[0]['result_retry']['pending']
        with self.assertRaisesRegex(AssertionError,'schema'):check(*a)
    def test_descriptor_changed_bytes_or_flags_rejected(self):
        for field,value in [('payload_sha256','0'*64),('attempts',1),('call_returned',False),('terminal_event_recorded',False)]:
            a=list(one_case());a[0]['result_retry']['last_result'][field]=value
            with self.subTest(field=field),self.assertRaises(AssertionError):check(*a)
    def test_malformed_result_digest_rejected(self):
        a=list(one_case());a[0]['events'][1]['result_sha256']='0'*64
        with self.assertRaisesRegex(AssertionError,'hash'):check(*a)
    def test_wrong_recorded_command_is_command_failure_with_consistent_bytes(self):
        a=list(one_case());w=a[0];old=w['events'][1]['result_sha256'];raw=json.loads(base64.b64decode(w['events'][1]['payload']))
        raw['command']['command_id']='different-recorded-command';newraw=encoded(raw);newsha=digest(newraw)
        for event in w['events']:
            for key in ('result_sha256','payload_sha256'):
                if event.get(key)==old:event[key]=newsha
            if event.get('reason')=='WORKER_RESULT_READY':event['payload']=b64(newraw)
        desc=w['result_retry']['last_result'];desc['payload']=b64(newraw);desc['payload_sha256']=newsha;a[-1]=[newsha]
        r=check(*a);self.assertEqual(r['recorded_reply_mismatches'],[1]);self.assertFalse(r['full_worker_coverage'])
    def test_write_then_unknown_tail_may_explain_received_bytes_without_return_credit(self):
        a=list(one_case(('PUBLISHED',)));w=a[0];state=w['result_retry'];desc=state['last_result']
        # No observer/attempt event returned, but a later plant receipt proves the
        # exact attempted bytes could have been committed before the exception.
        del w['events'][2:4]
        w['events'][-1].update(termination='WORKER_ITERATION_EXCEPTION',last_status=None,call_returned=False)
        desc.update(last_status=None,call_returned=False,returned_status_evidence=None,attempt_end_ns=None,
            attempt_event_recorded=False,terminal_reason='WORKER_ITERATION_EXCEPTION')
        state['counts'].update(iterations_completed=0,publications_returned=0,publications_published=0,publication_events=0)
        w['polls']=0;w['failure']={'type':'RuntimeError','detail':'unknown publish return'};state['failure']=w['failure'];state['pending']=copy.deepcopy(desc)
        r=check(*a);self.assertEqual(r['returned_publications'],0);self.assertEqual(r['published_publications'],0)
        self.assertEqual(r['plant_received_uncertain_payloads'],a[-1]);self.assertTrue(r['worker_failed'])
    def test_reserved_attempt_event_retains_known_published_credit(self):
        a=list(one_case(('PUBLISHED',)));w=a[0];s=w['result_retry'];d=s['last_result']
        attempt=w['events'].pop(3);w['events'].pop();w['overflow']=encoded(attempt).decode()
        d.update(attempt_event_recorded=False,terminal_event_recorded=False,terminal_reason='WORKER_ITERATION_EXCEPTION')
        s['counts'].update(iterations_completed=0,publication_events=0,terminal_events=0);w['polls']=0
        w['failure']={'type':'CapacityFailure','detail':'full'};s['failure']=w['failure'];s['pending']=copy.deepcopy(d)
        r=check(*a);self.assertEqual(r['published_publications'],1);self.assertTrue(r['worker_failed'])
    def test_no_worker_with_received_result_is_invalid(self):
        a=list(one_case());a[0]=None
        with self.assertRaises(AssertionError):check(*a)

    def test_published_late_preserves_known_success_without_qualification(self):
        a=list(one_case(('PUBLISHED',)));w=a[0];s=w['result_retry'];d=s['last_result']
        w['events'][2]['end_ns']=120000001;w['events'][3]['end_ns']=120000002
        w['events'][4].update(observed_ns=120000002,termination='PUBLISHED_LATE')
        d.update(attempt_end_ns=120000002,terminal_ns=120000002,terminal_reason='PUBLISHED_LATE')
        s['counts']['iterations_completed']=0;w['polls']=0;w['end_ns']=120000003
        w['failure']={'type':'WorkerResultFailure','detail':'PUBLISHED_LATE'};s['failure']=w['failure'];s['pending']=copy.deepcopy(d)
        r=check(*a);self.assertEqual(r['published_publications'],1);self.assertTrue(r['worker_failed']);self.assertFalse(r['full_worker_coverage'])

    def test_stop_pending_does_not_gain_extra_loop_or_retry(self):
        a=list(one_case(('BUSY',)));w=a[0];s=w['result_retry'];d=s['last_result']
        w['events'][-1].update(termination='STOP_WITH_OWNED_WORK',observed_ns=100000300)
        d.update(terminal_reason='STOP_WITH_OWNED_WORK',terminal_ns=100000300)
        s['counts'].update(iterations_started=1,iterations_completed=1);w['polls']=1;sync_last(w)
        r=check(*a);self.assertEqual(r['publication_attempts'],1);self.assertTrue(r['worker_failed'])

    def test_reply_exception_keeps_pre_reply_input_without_fake_output(self):
        a=list(one_case(('PUBLISHED',)));w=a[0];s=w['result_retry'];d=s['last_result'];a[-1]=[]
        w['events']=w['events'][:1]
        d.update(payload=None,payload_sha256=None,payload_ready_ns=None,ready_event_recorded=False,attempts=0,
            last_iteration=None,attempt_start_ns=None,attempt_end_ns=None,last_status=None,call_returned=False,
            returned_status_evidence=None,attempt_event_recorded=False,terminal_reason='WORKER_ITERATION_EXCEPTION',
            terminal_ns=100000100,terminal_event_recorded=True)
        w['events'].append(dict(reason='WORKER_RESULT_TERMINAL',activation=1,termination=d['terminal_reason'],observed_ns=d['terminal_ns'],
            deadline_ns=120000000,attempts=0,last_status=None,call_returned=False,job_sha256=d['source']['payload_sha256'],result_sha256=None))
        s['counts'].update(iterations_completed=0,replies_completed=0,ready_events=0,publications_attempted=0,
            publications_returned=0,publications_published=0,publication_events=0)
        w['polls']=0;w['failure']={'type':'RuntimeError','detail':'reply failed'};s['failure']=w['failure'];s['phase']='REPLY_CONSTRUCTION';s['pending']=copy.deepcopy(d)
        r=check(*a);self.assertEqual(r['computed_results'],0);self.assertEqual(r['publication_attempts'],0)

    def test_buffered_second_slot_survives_stop_without_second_reply(self):
        a=list(one_case(('BUSY',)));w=a[0];s=w['result_retry'];d=s['last_result'];old_id=a[1][1];old_job=a[2]['1']
        # The fixed slot0/key2 job is taken before slot1/key1 in the same poll.
        identity=copy.deepcopy(old_id);identity.update(activation=2,sequence=1,snapshot_control=1,snapshot_physics=10,deadline_ns=140000000)
        job=encoded(dict(identity,snapshot_payload=b64(b'second-owned-input')))
        command=copy.deepcopy(a[3][1]);command['command_id']='saved:2'
        payload=encoded(dict(identity=identity,command=command,completed_ns=100000100));newsha=digest(payload);oldsha=d['payload_sha256']
        for e in w['events']:
            if 'activation' in e:e['activation']=2
            if 'key' in e:e['key']=2
            if 'deadline_ns' in e:e['deadline_ns']=140000000
            for field in ('job_sha256','payload_sha256'):
                if e.get(field)==digest(old_job):e[field]=digest(job)
            for field in ('result_sha256','payload_sha256'):
                if e.get(field)==oldsha:e[field]=newsha
            if e.get('reason')=='WORKER_RESULT_READY':e.update(identity=identity,payload=b64(payload))
        w['events'][0]['slot']=0
        w['events'].insert(1,dict(direction='worker-jobs',operation='poll',slot=1,status='TAKEN',key=1,version=1,
            payload_sha256=digest(old_job),start_ns=100000010,end_ns=100000020))
        source=dict(slot=0,key=2,version=1,payload=b64(job),payload_sha256=digest(job))
        d.update(source=source,identity=identity,payload=b64(payload),payload_sha256=newsha,terminal_reason='STOP_WITH_OWNED_WORK',terminal_ns=100000300)
        w['events'][-1].update(termination='STOP_WITH_OWNED_WORK',observed_ns=100000300)
        s['counts'].update(jobs_taken=2,iterations_started=1,iterations_completed=1);w['polls']=1
        s['buffered_jobs']=[dict(slot=1,key=1,version=1,payload=b64(old_job),payload_sha256=digest(old_job))]
        s['pending']=copy.deepcopy(d);a[1][2]=identity;a[2]['2']=job;a[3].append(command)
        r=check(*a);self.assertEqual(r['buffered_jobs'],[1]);self.assertEqual(r['computed_results'],1)
        broken=copy.deepcopy(a);broken[0]['result_retry']['buffered_jobs']=[]
        with self.assertRaises(AssertionError):check(*broken)

    def test_failed_pending_cannot_be_silently_cleared(self):
        a=list(one_case(('BUSY',)));a[0]['result_retry']['pending']=None
        with self.assertRaisesRegex(AssertionError,'ownership'):check(*a)

    def test_extra_schema_field_and_erased_failure_rejected(self):
        a=list(one_case());a[0]['events'][1]['new_deadline_ns']=140000000
        with self.assertRaisesRegex(AssertionError,'schema'):check(*a)
        a=list(one_case(('BUSY',)));a[0]['result_retry']['failure']=None
        with self.assertRaisesRegex(AssertionError,'failure'):check(*a)


if __name__=='__main__':unittest.main()
