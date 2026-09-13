"""Independent saved-byte retry accounting. No producer import or side effects."""
import base64
import hashlib
import json


def digest(value):return hashlib.sha256(value).hexdigest()
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def integer(value,name):
    if type(value) is not int or value<0:raise AssertionError('Integer '+name)
    return value


def check(identities,jobs,events,transports,summary,epoch,arrays,published,
          event_overflow=None,transport_overflow=None,terminal_interruption=None):
    """Reconstruct every attempted/returned/recorded publication and pending tail.

    A final transport exception may leave no result, or a write whose observer did
    not return. The owned final descriptor retains that uncertainty. It is never
    counted as a returned publication or successful foundation publication.
    """
    overflow_event=json.loads(event_overflow) if event_overflow is not None else None
    overflow_transport=json.loads(transport_overflow) if transport_overflow is not None else None
    sequence=[(dict(v),True) for v in events if v['reason'] in ('JOB_PUBLICATION','PENDING_JOB_EXPIRED')]
    if overflow_event and overflow_event['reason'] in ('JOB_PUBLICATION','PENDING_JOB_EXPIRED'):
        sequence.append((overflow_event,False))
    trace=[dict(v,observer_record_committed=True) for v in transports
           if v['direction']=='plant-jobs' and v['operation']=='publish']
    if overflow_transport and overflow_transport['direction']=='plant-jobs' and overflow_transport['operation']=='publish':
        trace.append(dict(overflow_transport,observer_record_committed=False))
    last=summary['last_job_publication']
    failed=summary['failure'] is not None or terminal_interruption is not None
    if terminal_interruption is not None and (type(terminal_interruption) is not dict or not terminal_interruption):
        raise AssertionError('Bound outer interruption record required')
    native_returned=integer(summary['returned'],'actual returned native count')
    saved_ticks=len(arrays['step_actual_start'])
    if not saved_ticks<=native_returned<=saved_ticks+1:
        raise AssertionError('Saved/returned terminal tick coverage differs')
    pub_events=[v for v,_ in sequence if v['reason']=='JOB_PUBLICATION']
    if last is not None:
        k=last['identity']['activation'];ordinal=last['attempts']
        existing=[v for v in pub_events if v['activation']==k and v['attempt']==ordinal]
        if not existing:
            if not failed or last['event_recorded'] is not False:
                raise AssertionError('Missing publication event without exact failed-tail evidence')
            sequence.append((dict(reason='JOB_PUBLICATION',activation=k,physics=last['last_attempt_index'],
                attempt=ordinal,status=last['last_status'],deadline_ns=last['deadline_ns'],
                retry_checked_ns=last['retry_checked_ns'],input_digest=last['identity']['input_digest'],
                source_window=last['identity']['window_id'],payload_sha256=digest(base64.b64decode(last['payload'],validate=True))),False))
    by_job={};pending=None;last_expected=None;published_expected=set()
    attempted=returned=recorded=expired=transport_index=0;uncertain=[];uncommitted_success=[];previous_physics=-1;prior_transport_end=-1
    for event,committed in sequence:
        physics=integer(event['physics'],'publication physics')
        if physics>saved_ticks:raise AssertionError('Publication/expiry beyond actual terminal tick')
        if physics<previous_physics:raise AssertionError('Publication/expiry physics moved backward')
        previous_physics=physics
        k=integer(event['activation'],'activation')
        if k not in identities:raise AssertionError('Publication/expiry has no issued job')
        identity=identities[k];raw=jobs[str(k)];deadline=epoch+20_000_000*k
        if event['reason']=='PENDING_JOB_EXPIRED':
            if pending is None or pending['identity']['activation']!=k or pending['last_status']!='BUSY':
                raise AssertionError('Expired job was not pending BUSY')
            for field,wanted in [('attempts',pending['attempts']),('last_attempt_index',pending['last_attempt_index']),
                                 ('last_status','BUSY'),('deadline_ns',deadline),('payload_sha256',digest(raw))]:
                if type(event[field]) is not type(wanted) or event[field]!=wanted:raise AssertionError('Expiry identity/counter differs: '+field)
            now=integer(event['observed_ns'],'expiry check')
            if now<deadline or physics!=pending['last_attempt_index']+1:
                raise AssertionError('Expiry must be the immediately next eligible tick')
            if physics<len(arrays['step_actual_start']):
                if not int(arrays['step_actual_start'][physics])<=now<=int(arrays['step_actual_end'][physics]):
                    raise AssertionError('Expiry check outside actual step interval')
            elif not failed:raise AssertionError('Unexecuted expiry tick without failure')
            if event['expiration']=='ACTIVATION_BOUNDARY':
                if physics!=10*k:raise AssertionError('Expiry not at original activation boundary')
            elif event['expiration']=='ORIGINAL_DEADLINE':
                if physics%10==0 or physics>=10*k:raise AssertionError('Expiry not at overdue non-boundary retry')
            else:raise AssertionError('Unknown expiry cause')
            pending=None;expired+=1
            continue
        ordinal=integer(event['attempt'],'attempt ordinal')
        previous=by_job.get(k)
        if previous is None and pending is not None:
            raise AssertionError('New job overwrote unresolved pending publication')
        if not 1<=ordinal<=10 or ordinal!=(0 if previous is None else previous['attempts'])+1:
            raise AssertionError('Publication attempts skipped, duplicated or exceeded ten')
        if physics!=identity['snapshot_physics']+ordinal-1 or physics>=10*k:
            raise AssertionError('Publication not once per eligible predecessor tick')
        if previous is not None and (pending is None or pending['identity']['activation']!=k or previous['last_status']!='BUSY'):
            raise AssertionError('Retried a terminal/expired publication')
        for field,wanted in [('deadline_ns',deadline),('input_digest',identity['input_digest']),
                             ('source_window',identity['window_id']),('payload_sha256',digest(raw))]:
            if type(event[field]) is not type(wanted) or event[field]!=wanted:raise AssertionError('Publication identity differs: '+field)
        checked=event['retry_checked_ns']
        if ordinal==1:
            if checked is not None:raise AssertionError('Initial publication gained a retry guard')
        elif integer(checked,'retry check')>=deadline:raise AssertionError('Retry check at/after original deadline')
        status=event['status']
        if status is not None and (type(status) is not str or not status):raise AssertionError('Publication status schema')
        if status is None and (committed or not failed):raise AssertionError('Unknown observer return not preserved as failure')
        transport=trace[transport_index] if transport_index<len(trace) else None
        if transport is None:
            if status is not None or not failed:raise AssertionError('Publication lacks transport record')
        else:
            if transport['key']!=k or transport['payload_sha256']!=digest(raw):raise AssertionError('Transport key/payload changed or reordered')
            start=integer(transport['start_ns'],'transport start');end=integer(transport['end_ns'],'transport end')
            if end<start or start<identity['created_ns'] or start<prior_transport_end:raise AssertionError('Transport time order')
            prior_transport_end=end
            if ordinal>1 and start<checked:raise AssertionError('Transport precedes retry check')
            if status is not None and (transport['status']!=status or transport['observer_record_committed'] is not True):
                raise AssertionError('Returned observer status differs from committed transport')
            if physics<len(arrays['step_actual_start']):
                wake=int(arrays['step_actual_start'][physics]);finish=int(arrays['step_actual_end'][physics])
                if start<wake or end>finish or (checked is not None and checked<wake):raise AssertionError('Publication/check outside actual step interval')
            elif not failed:raise AssertionError('Unexecuted publication tick without failure')
            transport_index+=1
        attempted+=1;returned+=status is not None;recorded+=committed
        if status is None:
            uncertain.append(dict(activation=k,physics=physics,transport_status=None if transport is None else transport['status']))
            all_transports=list(transports)+([overflow_transport] if overflow_transport is not None else [])
            if transport is not None:
                raw_transport={key:value for key,value in transport.items() if key!='observer_record_committed'}
                if not all_transports or encoded(all_transports[-1])!=encoded(raw_transport):
                    raise AssertionError('Transport continued after unresolved publication')
            else:
                last_before_call=identity['created_ns'] if checked is None else checked
                if any(integer(v['end_ns'],'prior transport end')>last_before_call for v in all_transports):
                    raise AssertionError('Unlogged unresolved publication has later transport')
            last_before_call=identity['created_ns'] if checked is None else checked
            if any('received_ns' in v and integer(v['received_ns'],'prior admission')>last_before_call for v in events):
                raise AssertionError('Admission continued after unresolved publication')
        descriptor=dict(identity=identity,payload=base64.b64encode(raw).decode('ascii'),deadline_ns=deadline,
            attempts=ordinal,last_attempt_index=physics,last_status=status,event_recorded=committed,retry_checked_ns=checked)
        by_job[k]=descriptor;last_expected=descriptor
        pending=descriptor if status in ('BUSY',None) else None
        if status=='PUBLISHED':
            published_expected.add(k)
            if not committed:uncommitted_success.append(k)
    if transport_index!=len(trace):raise AssertionError('Extra/unexplained publication transport')
    for name,wanted in [('job_publication_attempted',attempted),('job_publication_returned',returned),
                        ('job_publication_events',recorded),('pending_expirations',expired)]:
        if integer(summary[name],name)!=wanted:raise AssertionError('Publication counter differs: '+name)
    if encoded(summary['last_job_publication'])!=encoded(last_expected) or encoded(summary['pending_publication'])!=encoded(pending):
        raise AssertionError('Owned last/pending publication descriptor differs')
    if any(type(v) is not int or v<1 for v in published):raise AssertionError('Published activation integer schema')
    if set(published)!=published_expected:raise AssertionError('Published set includes unknown write or misses returned success')
    if len(uncertain)>1 or (uncertain and not failed):raise AssertionError('More than one unresolved terminal publication')
    if pending is not None:
        if pending['last_attempt_index'] not in (native_returned-1,native_returned):
            raise AssertionError('Pending tail omitted completed eligible retry ticks')
        if summary.get('input_fault') is not None:raise AssertionError('Pending job survived latched input fault')
    if uncertain:
        point=uncertain[0]['physics']
        if point!=native_returned or point!=saved_ticks:
            raise AssertionError('Unresolved observer return was not actual terminal pre-native tick')
        if any(integer(v['physics'],'foundation event physics')>point for v in events):
            raise AssertionError('Foundation continued after unresolved publication')
    return dict(attempted=attempted,returned=returned,recorded_events=recorded,expired=expired,
                retry_attempts=attempted-len(by_job),per_job_attempts={str(k):v['attempts'] for k,v in by_job.items()},
                unresolved_observer_returns=uncertain,pending_present=pending is not None,
                successful_publications_without_committed_event=uncommitted_success,
                fixed_deadline_checks_exact=True,published_set_exact=True,terminal_tick_coverage_exact=True,
                bound_outer_interruption_present=terminal_interruption is not None,
                full_publication_coverage=published_expected==set(range(1,1819)) and not uncertain and pending is None)
