"""Independent saved worker/result reconstruction; no producer/runtime imports."""
import base64,hashlib,json

CONTRACT='immutable_result_BUSY_max20_original_deadline'
COUNTERS=('iterations_started','iterations_completed','polls_attempted','polls_returned',
 'jobs_taken','jobs_decoded','replies_attempted','replies_completed','ready_events',
 'publications_attempted','publications_returned','publications_published','publications_BUSY',
 'publication_events','terminals','terminal_events')
DESCRIPTOR={'source','received_ns','identity','completed_ns','payload','payload_sha256','payload_ready_ns',
 'ready_event_recorded','attempts','last_iteration','attempt_start_ns','attempt_end_ns','last_status',
 'call_returned','returned_status_evidence','attempt_event_recorded','terminal_reason','terminal_ns','terminal_event_recorded'}

def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(v).hexdigest()
def integer(v,name):
    if type(v) is not int or v<0:raise AssertionError('Nonnegative integer '+name)
    return v
def raw64(text):
    if type(text) is not str:raise AssertionError('Literal base64 text')
    return base64.b64decode(text,validate=True)
def same(a,b,name):
    if encoded(a)!=encoded(b):raise AssertionError(name)
def optional_ns(v,name):return None if v is None else integer(v,name)


def _source(v,jobs):
    if type(v) is not dict or set(v)!={'slot','key','version','payload','payload_sha256'}:raise AssertionError('Owned job source schema')
    k=integer(v['key'],'owned job key');slot=integer(v['slot'],'owned job slot')
    if slot>=2 or integer(v['version'],'owned job version')==0 or str(k) not in jobs:raise AssertionError('Owned job source identity')
    raw=raw64(v['payload'])
    if raw!=jobs[str(k)] or v['payload_sha256']!=digest(raw):raise AssertionError('Owned job bytes differ')
    return k


def _descriptor(v,identities,jobs):
    if v is None:return None
    if type(v) is not dict or set(v)!=DESCRIPTOR:raise AssertionError('Pending/last descriptor schema')
    k=_source(v['source'],jobs);integer(v['received_ns'],'received')
    for name in ('ready_event_recorded','call_returned','attempt_event_recorded','terminal_event_recorded'):
        if type(v[name]) is not bool:raise AssertionError('Literal descriptor boolean')
    for name in ('completed_ns','payload_ready_ns','last_iteration','attempt_start_ns','attempt_end_ns','terminal_ns'):
        optional_ns(v[name],name)
    if v['identity'] is not None:same(v['identity'],identities[k],'Descriptor issued identity differs')
    if not 0<=integer(v['attempts'],'attempts')<=20:raise AssertionError('Twenty-result-attempt cap')
    if v['last_status'] is not None and type(v['last_status']) is not str:raise AssertionError('Descriptor status schema')
    if v['terminal_reason'] is not None and type(v['terminal_reason']) is not str:raise AssertionError('Terminal reason schema')
    if v['payload'] is None:
        if v['payload_sha256'] is not None or v['payload_ready_ns'] is not None or v['ready_event_recorded'] or v['attempts']:
            raise AssertionError('Unreturned reply granted payload/attempt credit')
    elif digest(raw64(v['payload']))!=v['payload_sha256']:raise AssertionError('Descriptor result digest')
    if v['attempts']==0:
        if any(v[name] is not None for name in ('last_iteration','attempt_start_ns','attempt_end_ns','last_status','returned_status_evidence')) or v['call_returned'] or v['attempt_event_recorded']:
            raise AssertionError('Zero-attempt descriptor gained call evidence')
    elif v['attempt_start_ns'] is None or v['last_iteration'] is None:raise AssertionError('Attempt lacks start/iteration')
    if not v['call_returned'] and (v['last_status'] is not None or v['returned_status_evidence'] is not None or v['attempt_end_ns'] is not None or v['attempt_event_recorded']):
        raise AssertionError('Unknown return gained known status/end/event')
    if v['call_returned'] and v['last_status'] is not None:
        same(json.loads(v['returned_status_evidence']),dict(type='builtins.str',value=v['last_status']),'Returned status evidence differs')
    elif v['call_returned'] and (type(v['returned_status_evidence']) is not str or type(json.loads(v['returned_status_evidence'])) is not dict):
        raise AssertionError('Invalid returned status lacks owned evidence')
    if v['terminal_event_recorded'] and v['terminal_reason'] is None:raise AssertionError('Terminal event without terminal state')
    return k


def check(worker,identities,jobs,commands,epoch,schedule,plant_hashes):
    if worker is None:
        if plant_hashes:raise AssertionError('Plant result lacks worker evidence')
        return dict(worker_report_present=False,recorded_reply_mismatches=[],full_worker_coverage=False)
    if worker['recorded_command_table_sha256']!=schedule or worker['model_calls']!=0 or worker['native_steps']!=0:raise AssertionError('Worker table/scope changed')
    for name in ('pid','polls','start_ns','end_ns'):integer(worker[name],name)
    if worker['pid']==0 or worker['end_ns']<worker['start_ns'] or worker['polls']>60000:raise AssertionError('Worker lifecycle bounds')
    state=worker['result_retry']
    if set(state)!={'contract','max_attempts','counts','phase','closed','pending','last_result','buffered_jobs','last_poll_return','clock_failure','failure'}:
        raise AssertionError('Worker retry output schema')
    if state['contract']!=CONTRACT or type(state['max_attempts']) is not int or state['max_attempts']!=20 or state['closed'] is not True:raise AssertionError('Worker retry contract/final close')
    if type(state['phase']) is not str or state['phase'] not in ('READY','JOB_POLL','JOB_DECODE','REPLY_CONSTRUCTION','RESULT_PUBLICATION'):raise AssertionError('Worker phase')
    counts=state['counts']
    if set(counts)!=set(COUNTERS):raise AssertionError('Worker counter fields')
    for name in COUNTERS:integer(counts[name],name)
    started,completed=counts['iterations_started'],counts['iterations_completed']
    failed=worker['failure'] is not None or state['failure'] is not None or worker['overflow'] is not None
    if (worker['failure'] is None)!=(state['failure'] is None):raise AssertionError('Worker/state first failure was erased')
    if worker['overflow'] is not None and state['failure'] is None:raise AssertionError('Logger overflow lacks worker failure')
    if started not in (completed,completed+1) or completed!=worker['polls'] or started>60000:raise AssertionError('Worker loop attempt/completion accounting')
    if not failed and started!=completed:raise AssertionError('Incomplete iteration without failure')
    if type(state['buffered_jobs']) is not list or len(state['buffered_jobs'])>2:raise AssertionError('Fixed buffered-job bound')
    buffered_keys=[_source(v,jobs) for v in state['buffered_jobs']]
    last=state['last_result'];last_key=_descriptor(last,identities,jobs)
    pending=state['pending'];pending_key=_descriptor(pending,identities,jobs)
    if last is not None:
        retained=last['terminal_reason']!='PUBLISHED' or not last['terminal_event_recorded']
        if retained!=(pending is not None):raise AssertionError('Final pending ownership differs from terminal state')
    if pending is not None:
        same(pending,last,'Active pending/last descriptor differs')
        if not failed or pending['terminal_reason'] is None:raise AssertionError('Final pending work must be failed and terminal')
    if buffered_keys and not failed:raise AssertionError('Buffered jobs hidden by successful worker')
    if state['clock_failure'] is not None and (not failed or type(state['clock_failure']) is not dict):raise AssertionError('Clock failure erased')
    for name in ('last_poll_return',):
        if state[name] is not None and type(json.loads(state[name])) is not dict:raise AssertionError('Owned poll representation schema')
    entries=[(i*4,dict(v),True) for i,v in enumerate(worker['events'])]
    if worker['overflow'] is not None:
        overflow=json.loads(worker['overflow'])
        if type(overflow) is not dict:raise AssertionError('Reserved worker overflow schema')
        entries.append((len(worker['events'])*4,overflow,False))
    ready={};pubs={};terminals={};takes=[];transport_pubs=[];poll_rows=[]
    previous_transport_end=worker['start_ns'];previous_transport=None
    for index,row,committed in entries:
        if 'operation' in row:
            start=integer(row['start_ns'],'transport start');end=integer(row['end_ns'],'transport end')
            sibling=(previous_transport is not None and row.get('direction')==previous_transport.get('direction')=='worker-jobs'
                and row.get('operation')==previous_transport.get('operation')=='poll'
                and (start,end)==(previous_transport['start_ns'],previous_transport['end_ns'])
                and row.get('slot')!=previous_transport.get('slot'))
            if sibling and row['slot']<previous_transport['slot']:raise AssertionError('Fixed poll slot order changed')
            if not (start<=end<=worker['end_ns'] and (previous_transport_end<=start or sibling)):raise AssertionError('Worker transport time order')
            previous_transport_end=end
            previous_transport=row
            if row['direction']=='worker-jobs' and row['operation']=='poll':
                if set(row)!={'direction','operation','slot','status','key','version','payload_sha256','start_ns','end_ns'}:raise AssertionError('Worker poll event schema')
                poll_rows.append((index,row,committed))
                if row['status']=='EMPTY':raise AssertionError('Unlogged empty poll fabricated')
                if row['status']=='TAKEN':
                    k=integer(row['key'],'taken key')
                    if k not in identities or row['payload_sha256']!=digest(jobs[str(k)]) or not 0<=integer(row['slot'],'slot')<2 or integer(row['version'],'version')==0:
                        raise AssertionError('Taken job key/bytes/slot/version')
                    takes.append((index,row,committed))
            elif row['direction']=='worker-results' and row['operation']=='publish':
                if set(row)!={'direction','operation','key','status','payload_sha256','start_ns','end_ns'}:raise AssertionError('Worker publish event schema')
                transport_pubs.append((index,row,committed))
            else:raise AssertionError('Unknown worker transport operation')
        else:
            reason=row.get('reason');k=integer(row['activation'],'worker activation')
            if k not in identities:raise AssertionError('Worker event lacks issued job')
            if reason=='WORKER_RESULT_READY':
                if set(row)!={'reason','activation','identity','received_ns','completed_ns','payload_ready_ns','job_sha256','result_sha256','payload'}:raise AssertionError('Worker ready event schema')
                if k in ready:raise AssertionError('Result recomputed/duplicate ready')
                ready[k]=(index,row,committed)
            elif reason=='WORKER_RESULT_PUBLICATION':
                if set(row)!={'reason','activation','iteration','attempt','start_ns','end_ns','status','deadline_ns','job_sha256','result_sha256'}:raise AssertionError('Worker attempt event schema')
                pubs.setdefault(k,[]).append((index,row,committed))
            elif reason=='WORKER_RESULT_TERMINAL':
                if set(row)!={'reason','activation','termination','observed_ns','deadline_ns','attempts','last_status','call_returned','job_sha256','result_sha256'}:raise AssertionError('Worker terminal event schema')
                if k in terminals:raise AssertionError('Duplicate worker terminal')
                terminals[k]=(index,row,committed)
            else:raise AssertionError('Unknown worker result event')
    # Only the final owned result can supply uncommitted pre-error observations.
    if last is not None:
        k=last_key
        end_index=(terminals[k][0] if k in terminals else (len(worker['events'])+1)*4)
        if k not in ready and last['payload'] is not None:
            if not failed or last['ready_event_recorded']:raise AssertionError('Missing ready event without failed descriptor')
            ready[k]=(end_index-2,dict(activation=k,identity=last['identity'],received_ns=last['received_ns'],
                completed_ns=last['completed_ns'],payload_ready_ns=last['payload_ready_ns'],job_sha256=digest(jobs[str(k)]),
                result_sha256=last['payload_sha256'],payload=last['payload']),False)
        if last['attempts']>len(pubs.get(k,[])):
            if not failed or last['attempt_event_recorded'] or last['attempts']!=len(pubs.get(k,[]))+1:raise AssertionError('Missing attempt is not one failed tail')
            pubs.setdefault(k,[]).append((end_index-1,dict(activation=k,iteration=last['last_iteration'],attempt=last['attempts'],
                start_ns=last['attempt_start_ns'],end_ns=last['attempt_end_ns'],status=last['last_status'],
                deadline_ns=identities[k]['deadline_ns'],job_sha256=digest(jobs[str(k)]),result_sha256=last['payload_sha256'],
                _call_returned=last['call_returned']),False))
        if last['terminal_reason'] is not None and k not in terminals:
            if not failed or last['terminal_event_recorded']:raise AssertionError('Missing terminal without failed descriptor')
            terminals[k]=(end_index,dict(activation=k,termination=last['terminal_reason'],observed_ns=last['terminal_ns'],
                deadline_ns=None if last['identity'] is None else identities[k]['deadline_ns'],attempts=last['attempts'],
                last_status=last['last_status'],call_returned=last['call_returned'],job_sha256=digest(jobs[str(k)]),
                result_sha256=last['payload_sha256']),False)
    keys=set(ready)|set(pubs)|set(terminals)
    if last is not None:keys.add(last_key)
    rows={};mismatches=[];successful=[];unknown=[];used_transport=set();used_takes=set()
    returned=published=busy=attempted=0;all_iterations=[]
    for k in keys:
        identity=identities[k];deadline=epoch+20_000_000*k
        if type(identity.get('deadline_ns')) is not int or identity['deadline_ns']!=deadline:raise AssertionError('Worker original deadline differs')
        candidates=[x for x in takes if x[1]['key']==k]
        if len(candidates)!=1:raise AssertionError('Prepared result lacks unique taken job')
        take_index,take,take_committed=candidates[0];used_takes.add(take_index)
        if take['end_ns']<identity['created_ns']:raise AssertionError('Worker took job before it existed')
        item=ready.get(k);descriptor=last if k==last_key else None
        if item:
            index,row,committed=item
            same(row['identity'],identity,'Worker ready issued identity changed')
            if row['job_sha256']!=digest(jobs[str(k)]):raise AssertionError('Worker ready job digest')
            received=integer(row['received_ns'],'ready receive');completion=integer(row['completed_ns'],'completion')
            ready_ns=optional_ns(row['payload_ready_ns'],'payload ready')
            if not identity['created_ns']<=take['end_ns']<=received<=completion:raise AssertionError('Worker exact job/timestamp binding')
            if completion>=deadline:raise AssertionError('Reply computed after known original deadline')
            if ready_ns is None:
                if not failed or committed or k!=last_key or state['clock_failure'] is None:raise AssertionError('Missing ready clock without preserved clock failure')
            elif not completion<=ready_ns<=worker['end_ns']:raise AssertionError('Payload ready clock order')
            raw=raw64(row['payload']);parsed=json.loads(raw)
            if set(parsed)!={'identity','command','completed_ns'} or raw!=encoded(parsed):raise AssertionError('Result canonical fields/bytes')
            same(parsed['identity'],identity,'Result identity changed');same(parsed['completed_ns'],completion,'Result completion changed')
            if digest(raw)!=row['result_sha256']:raise AssertionError('Ready payload hash differs')
            expected=encoded(dict(identity=identity,command=commands[k],completed_ns=completion))
            if raw!=expected:mismatches.append(k)
        else:
            if descriptor is None or not failed or descriptor['ready_event_recorded'] or descriptor['payload'] is not None:raise AssertionError('Uncomputed job is not final failed owned input')
            received=descriptor['received_ns'];completion=descriptor['completed_ns'];ready_ns=None;raw=None;index=terminals.get(k,((len(worker['events'])+1)*4,None,None))[0]-1
            if not take['end_ns']<=received<=worker['end_ns']:raise AssertionError('Owned pre-reply input clock')
        if take_index>=index:raise AssertionError('Prepared before taking job')
        attempts=pubs.get(k,[]);last_iteration=None;last_end=ready_ns;statuses=[]
        for ordinal,(pi,row,committed) in enumerate(attempts,1):
            if raw is None or ready_ns is None or row['attempt']!=ordinal or type(row['attempt']) is not int or ordinal>20:raise AssertionError('Result attempt ordinal/cap/ready credit')
            iteration=integer(row['iteration'],'publication iteration');all_iterations.append(iteration)
            if iteration>=started or (last_iteration is not None and iteration!=last_iteration+1):raise AssertionError('Pending retries must use each next worker iteration')
            if ordinal>1 and statuses[-1]!='BUSY':raise AssertionError('Only BUSY can be retried')
            start=integer(row['start_ns'],'attempt start');end=optional_ns(row['end_ns'],'attempt end');status=row['status']
            returned_call=row.get('_call_returned',True)
            if type(returned_call) is not bool or (status is not None and type(status) is not str):raise AssertionError('Attempt status/return schema')
            if not ready_ns<=start<deadline or (last_end is not None and start<last_end) or (end is not None and not start<=end<=worker['end_ns']):raise AssertionError('Original deadline or attempt clock order')
            if row['deadline_ns']!=deadline or row['job_sha256']!=digest(jobs[str(k)]) or row['result_sha256']!=digest(raw):raise AssertionError('Retry changed immutable result/job/deadline')
            matches=[x for x in transport_pubs if x[0] not in used_transport and x[1]['key']==k and x[1]['payload_sha256']==digest(raw)]
            transport=matches[0] if matches else None
            if transport is not None:
                ti,tr,tc=transport;used_transport.add(ti)
                if ti>=pi or tr['start_ns']<start or (end is not None and tr['end_ns']>end):raise AssertionError('Transport not inside actual attempt')
                if returned_call and (not tc or tr['status']!=status):raise AssertionError('Returned observer differs from committed transport')
            elif returned_call:raise AssertionError('Returned attempt lacks transport')
            if not returned_call:
                if not failed or committed or k!=last_key or ordinal!=len(attempts) or status is not None or end is not None:raise AssertionError('Unknown return not final failed attempt')
                unknown.append(digest(raw))
            elif end is None and (not failed or committed or k!=last_key or (state['clock_failure'] is None and status is not None)):raise AssertionError('Missing post-return clock unexplained')
            attempted+=1;returned+=returned_call;published+=returned_call and status=='PUBLISHED';busy+=returned_call and status=='BUSY'
            if returned_call and status=='PUBLISHED':successful.append(digest(raw))
            statuses.append(status);last_iteration=iteration;last_end=end
        terminal=terminals.get(k)
        if terminal is None:raise AssertionError('Final worker output lost terminal owned job')
        terminal_index,tr,tc=terminal;reason=tr['termination'];observed=optional_ns(tr['observed_ns'],'terminal clock')
        last_status=statuses[-1] if statuses else None;last_returned=attempts[-1][1].get('_call_returned',True) if attempts else False
        for field,wanted in [('attempts',len(attempts)),('last_status',last_status),('call_returned',last_returned),('job_sha256',digest(jobs[str(k)])),('result_sha256',None if raw is None else digest(raw))]:
            same(tr[field],wanted,'Terminal state/counter differs: '+field)
        expected_deadline=None if descriptor is not None and descriptor['identity'] is None else deadline
        same(tr['deadline_ns'],expected_deadline,'Terminal deadline differs')
        if observed is not None and (observed<received or observed>worker['end_ns']):raise AssertionError('Terminal clock bounds')
        if attempts:
            pi,tail,_=attempts[-1]
            if terminal_index<=pi:raise AssertionError('Terminal preceded attempt')
            end=tail['end_ns']
            if end is not None and observed is not None and observed<end:raise AssertionError('Terminal before returned attempt')
            if last_returned and end is not None:
                required=('PUBLISHED' if end<deadline else 'PUBLISHED_LATE') if last_status=='PUBLISHED' else ('PUBLICATION_'+last_status if last_status!='BUSY' else ('ORIGINAL_DEADLINE' if end>=deadline else ('ATTEMPT_LIMIT' if len(attempts)==20 else None)))
            else:required=None
            if required is not None and reason!=required:
                # Logging/clock failure may terminalize the already-returned call
                # before normal status dispatch, and its event flag proves that.
                if not (failed and k==last_key and reason=='WORKER_ITERATION_EXCEPTION' and (not descriptor['attempt_event_recorded'] or end is None)):
                    raise AssertionError('Result terminal reason differs from returned outcome')
            if required is None and reason not in ('ORIGINAL_DEADLINE','STOP_WITH_OWNED_WORK','WORKER_ITERATION_EXCEPTION','WORKER_EXCEPTION'):
                raise AssertionError('Unexpected pending-result termination')
        else:
            if reason not in ('EXPIRED_BEFORE_REPLY','ORIGINAL_DEADLINE','WORKER_ITERATION_EXCEPTION','WORKER_EXCEPTION','STOP_WITH_OWNED_WORK'):raise AssertionError('Unexpected pre-attempt terminal')
        if reason in ('ORIGINAL_DEADLINE','EXPIRED_BEFORE_REPLY') and (observed is None or observed<deadline):raise AssertionError('Expiry before original deadline')
        if reason!='PUBLISHED' and not failed:raise AssertionError('Failed terminal hidden as healthy worker')
        if k==last_key:
            if descriptor['attempts']!=len(attempts):raise AssertionError('Last descriptor attempt coverage')
            same(descriptor['terminal_reason'],reason,'Last terminal differs');same(descriptor['terminal_ns'],observed,'Last terminal time differs')
            same(descriptor['terminal_event_recorded'],tc,'Last terminal committed flag differs')
            same(descriptor['ready_event_recorded'],False if item is None else item[2],'Last ready committed flag differs')
            if item:
                for field in ('identity','received_ns','completed_ns','payload_ready_ns','payload'):
                    same(descriptor[field],item[1][field],'Last ready descriptor differs: '+field)
            if attempts:
                tail=attempts[-1]
                for field,event_field in [('last_iteration','iteration'),('attempt_start_ns','start_ns'),('attempt_end_ns','end_ns'),('last_status','status')]:same(descriptor[field],tail[1][event_field],'Last attempt descriptor differs')
                same(descriptor['attempt_event_recorded'],tail[2],'Last attempt committed flag')
                same(descriptor['call_returned'],last_returned,'Last known return flag')
        rows[k]=dict(take_index=take_index,take=take,ready_index=index,terminal_index=terminal_index,reason=reason,
            attempts=len(attempts),last_iteration=last_iteration,ready=item is not None,descriptor=descriptor)
    if len(set(all_iterations))!=len(all_iterations):raise AssertionError('More than one result publication in worker iteration')
    if len(used_transport)!=len(transport_pubs):raise AssertionError('Worker publication missing attempted-state accounting')
    if not set(plant_hashes)<=set(successful)|set(unknown):raise AssertionError('Plant result lacks possible published worker bytes')
    ordered=sorted(rows,key=lambda k:rows[k]['take_index'])
    for i,k in enumerate(ordered):
        row=rows[k]
        if i and (rows[ordered[i-1]]['reason']!='PUBLISHED' or row['ready_index']<=rows[ordered[i-1]]['terminal_index']):raise AssertionError('New reply before prior successful terminal')
        group=(row['take']['start_ns'],row['take']['end_ns'])
        for pi,poll,_ in poll_rows:
            if row['take_index']<pi<row['terminal_index'] and (pi>row['ready_index'] or (poll['start_ns'],poll['end_ns'])!=group):
                raise AssertionError('New-job polling while result pending')
        if row['reason']!='PUBLISHED' and any(index>row['terminal_index'] for index,_,_ in entries):raise AssertionError('Worker continued after terminal failure')
    acknowledged=[]
    for entry in takes:
        if entry[0] in used_takes:acknowledged.append(entry);continue
        matches=[v for v in state['buffered_jobs'] if v['key']==entry[1]['key'] and v['slot']==entry[1]['slot'] and v['version']==entry[1]['version']]
        if len(matches)==1:acknowledged.append(entry)
        elif not (failed and state['phase']=='JOB_POLL' and counts['polls_attempted']==counts['polls_returned']+1):raise AssertionError('Taken job silently dropped')
    if len(acknowledged)!=counts['jobs_taken'] or len(acknowledged)!=len(rows)+len(buffered_keys):raise AssertionError('Owned/buffered taken-job count differs')
    expected=dict(jobs_taken=len(acknowledged),jobs_decoded=len(ready),replies_attempted=len(ready),replies_completed=len(ready),
        ready_events=sum(v[2] for v in ready.values()),publications_attempted=attempted,publications_returned=returned,
        publications_published=published,publications_BUSY=busy,publication_events=sum(v[2] for values in pubs.values() for v in values),
        terminals=len(terminals),terminal_events=sum(v[2] for v in terminals.values()))
    if last_key is not None and last_key not in ready:
        expected['jobs_decoded']+=last['identity'] is not None
        expected['replies_attempted']+=last['completed_ns'] is not None
    for name,value in expected.items():
        if counts[name]!=value:raise AssertionError('Worker counter differs: '+name)
    retry_loops=sum(max(0,len(v)-1) for v in pubs.values())
    seen_groups=set();buffered_prepares=0
    for k in ordered:
        group=(rows[k]['take']['start_ns'],rows[k]['take']['end_ns'])
        if group in seen_groups:buffered_prepares+=1
        seen_groups.add(group)
    final_pending_no_attempt=bool(pending is not None and pending['attempts'] and pending['last_iteration']==started-2)
    expected_polls=started-retry_loops-buffered_prepares-int(final_pending_no_attempt)
    if counts['polls_attempted']!=expected_polls:raise AssertionError('Mailbox polls/worker-loop ownership differs')
    if not counts['polls_returned']<=counts['polls_attempted']<=counts['polls_returned']+1:raise AssertionError('Poll attempted/returned gap')
    if counts['polls_attempted']!=counts['polls_returned'] and not (failed and state['phase']=='JOB_POLL'):raise AssertionError('Unknown poll return is not terminal')
    if not failed and (pending is not None or state['failure'] is not None):raise AssertionError('Healthy worker retains failed state')
    full=not failed and len(ready)==len(rows)==1818 and published==1818 and not mismatches and not buffered_keys and pending is None
    return dict(worker_report_present=True,recorded_reply_mismatches=sorted(mismatches),full_worker_coverage=full,
        computed_results=len(ready),publication_attempts=attempted,returned_publications=returned,published_publications=published,
        BUSY_returns=busy,retry_attempts=retry_loops,terminals=len(terminals),buffered_jobs=buffered_keys,
        uncertain_publication_payloads=unknown,plant_received_uncertain_payloads=sorted(set(plant_hashes)&set(unknown)),
        counters_exact=True,worker_failed=failed,empty_poll_spans_not_logged=True,
        empty_poll_counts_crosschecked_against_source_loop_accounting=True)
