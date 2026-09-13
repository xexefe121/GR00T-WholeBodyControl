"""Independent canonical job/result/admission reconstruction from saved bytes."""
import base64,hashlib,json
import numpy as np
from clock_saved_math import exact
from retry_publication_math import check as check_publications


def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def b64(raw):return base64.b64encode(raw).decode('ascii')


def parsed(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise AssertionError('Duplicate JSON field')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique)


def command_wire(control,origin,target,action):
    return dict(command_id='saved:'+str(control)+':'+origin,origin='recorded-query250:'+origin,
                target=b64(target.tobytes()),raw_action=b64(action.tobytes()))


def canonical_commands(full,hold,full_sha,hold_sha):
    targets=np.concatenate([full['target'],hold['target']]);actions=np.concatenate([full['action'],hold['action']])
    frames=list(range(11,1580))+[1579]*250
    if targets.shape!=(1819,23) or targets.dtype!=np.float64 or actions.shape!=(1819,23) or actions.dtype!=np.float32:
        raise AssertionError('Original recorded command schema')
    commands=[command_wire(i,full_sha if i<1569 else hold_sha,targets[i],actions[i]) for i in range(1819)]
    schedule=digest(encoded(dict(commands=commands,frames=frames)))
    windows=['recorded:'+str(i)+':frame:'+str(frames[i])+':'+schedule for i in range(1819)]
    return commands,frames,schedule,windows


def job_expected(raw,a,metadata,binding,commands,windows):
    job=parsed(raw);c=job['snapshot_control'];activation=job['activation']
    if type(c) is not int or not 0<=c<len(a['control_control']) or activation!=c+1 or activation>=1819:
        raise AssertionError('Issued job control/activation')
    created=job['created_ns']
    if type(created) is not int or created<int(a['control_actual_activation_ns'][c]):
        raise AssertionError('Job created before actual snapshot/history boundary')
    actual=commands[next(i for i,v in enumerate(commands) if v['command_id']==metadata['control_command_ids'][c])]
    # Independent control-history helper has already reconstructed these fields.
    body=dict(state=b64(a['control_integration'][c].tobytes()),incoming_raw=b64(a['control_incoming_raw'][c].tobytes()),
        active_command=dict(actual,target=b64(a['control_target'][c].tobytes()),raw_action=b64(a['control_raw_action'][c].tobytes())),
        history_before=[(k,b64(v)) for k,v in metadata['control_history_before'][c]],
        terms=[(k,b64(v)) for k,v in metadata['control_measured_terms'][c]],
        history_after=[(k,b64(v)) for k,v in metadata['control_history_after'][c]])
    snapshot=encoded(body)
    identity=dict(binding=binding,sequence=c,snapshot_control=c,snapshot_physics=10*c,activation=activation,
        window_id=windows[activation],input_digest=digest(snapshot),history_digest=digest(a['control_flat_history_before'][c].tobytes()),
        schedule_digest=digest(encoded(body['active_command'])),created_ns=created)
    expected=dict(identity,snapshot_payload=b64(snapshot))
    if raw!=encoded(expected):raise AssertionError('Issued canonical job bytes differ from actual snapshot/history')
    return identity


def admission_reason(result,key,issued,published,seen_activations,seen_command_ids,binding,held_latched,received,epoch,limits):
    identity=result['identity'];command=result['command'];activation=identity['activation']
    if held_latched:return 'FAULT_LATCHED'
    if type(activation) is not int or key!=activation:return 'WRONG_ACTIVATION'
    if encoded(identity.get('binding'))!=encoded(binding):return 'WRONG_BINDING_OR_EPOCH'
    if activation not in issued:return 'UNKNOWN_JOB'
    if activation not in published:return 'JOB_NOT_PUBLISHED'
    if encoded(identity)!=encoded(issued[activation]):return 'JOB_OR_INPUT_MISMATCH'
    if activation in seen_activations or command['command_id'] in seen_command_ids:return 'DUPLICATE'
    if any(type(command.get(k)) is not str or not 0<len(command[k])<=128 for k in ('command_id','origin')):
        raise ValueError('malformed command identity')
    target=np.frombuffer(base64.b64decode(command['target'],validate=True),np.float64)
    action=np.frombuffer(base64.b64decode(command['raw_action'],validate=True),np.float32)
    if target.shape!=(23,) or action.shape!=(23,) or not np.isfinite(target).all() or not np.isfinite(action).all():raise ValueError('malformed command bytes')
    if np.any(target<limits[:,0]) or np.any(target>limits[:,1]):raise ValueError('out-of-native-range command')
    completed=result['completed_ns']
    if type(completed) is not int:raise ValueError('malformed completion time')
    if received>=epoch+activation*20000000:return 'LATE'
    if completed>received:return 'FUTURE_WORKER_TIMESTAMP'
    if completed<identity['created_ns']:return 'WORKER_TIMESTAMP_BEFORE_JOB'
    return None


def parsed_result(raw):
    result=parsed(raw)
    if not isinstance(result['identity'],dict):raise ValueError('result identity must be an object')
    command=result['command']
    for key in ('target','raw_action'):base64.b64decode(command[key],validate=True)
    for key in ('command_id','origin'):command[key]
    result['completed_ns']
    return result


def protocol(a,metadata,jobs,published_list,worker,request,full,hold,full_sha,hold_sha,reference_sha,contract,terminal_interruption=None):
    commands,frames,schedule,windows=canonical_commands(full,hold,full_sha,hold_sha)
    if schedule!=request['command_table_sha256']:raise AssertionError('Original table digest differs')
    binding=dict(run=request['run_id'],model_hash=request['expected_model_sha256'],reference_hash=reference_sha,epoch=request['input_epoch'])
    epoch=metadata['summary']['foundation']['epoch_ns'];c=len(a['control_control'])
    exact(a['control_nominal_source_frame'],np.asarray(frames[:c],np.int64),'Protocol source index')
    if metadata['control_nominal_windows']!=windows[:c]:raise AssertionError('Literal source windows')
    active_indices=[]
    for i,identifier in enumerate(metadata['control_command_ids']):
        matches=[k for k,v in enumerate(commands) if v['command_id']==identifier]
        if len(matches)!=1:raise AssertionError('Unknown actual recorded command identity')
        active_indices.append(matches[0])
        if metadata['control_active_windows'][i]!=windows[matches[0]]:raise AssertionError('Active source window does not match actual command')
    exact(a['control_active_source_frame'],np.asarray([frames[i] for i in active_indices],np.int64),'Actually active frames')
    if c and (active_indices[0]!=0 or not 0<=int(a['control_admitted_ns'][0])<epoch):
        raise AssertionError('Initial command must be admitted before the fixed epoch')
    identities={int(k):job_expected(raw,a,metadata,binding,commands,windows) for k,raw in jobs.items()}
    if any(str(k) not in jobs or identities[k]['activation']!=k for k in identities):raise AssertionError('Job dictionary key differs')
    expected_jobs=[i+1 for i in range(c) if i+1<1819 and not a['control_held'][i]]
    if list(sorted(identities))!=expected_jobs[:len(identities)] or len(expected_jobs)-len(identities)>1:
        raise AssertionError('Issued job coverage is not a preserved prefix')
    if published_list!=sorted(set(published_list)) or not set(published_list)<=set(identities):raise AssertionError('Published activation set differs')
    events=[parsed(raw) for raw in metadata['foundation_events']]
    transports=[parsed(raw) for raw in metadata['transport_records']]
    for event in transports:
        if type(event['start_ns']) is not int or type(event['end_ns']) is not int or event['end_ns']<event['start_ns']:
            raise AssertionError('Transport copy timestamp order')
    publication_audit=check_publications(identities,jobs,events,transports,metadata['summary']['foundation'],epoch,a,published_list,
        metadata.get('event_ledger_overflow'),metadata.get('transport_ledger_overflow'),terminal_interruption)
    issued_seen={};published_seen=set();sealed={};seen_ids={commands[0]['command_id']};latched=False
    result_hashes=[];rejections=[];held_events=[];publication_events=[]
    previous_physics=-1
    for event in events:
        physics=event['physics']
        if type(physics) is not int or physics<previous_physics:raise AssertionError('Foundation event physics order')
        previous_physics=physics;reason=event['reason']
        if reason=='JOB_PUBLICATION':
            k=event['activation'];identity=identities[k];issued_seen[k]=identity;publication_events.append(k)
            # Every retry identity, tick, guard, transport status and counter was
            # independently checked above. Admission still follows event order.
            if event['status']=='PUBLISHED':published_seen.add(k)
        elif reason=='PENDING_JOB_EXPIRED':
            pass  # Exact BUSY state, fixed deadline and counters checked above.
        elif reason=='HELD_COMMAND_INTERVAL':
            k=event['control'];latched=True;held_events.append(k)
            if k>=c or not a['control_held'][k] or physics!=10*k or event['command_id']!=metadata['control_command_ids'][k]:
                raise AssertionError('Held event does not match actual control')
            if event['nominal_window_id']!=windows[k] or event['active_window_id']!=metadata['control_active_windows'][k]:raise AssertionError('Held source gap hidden')
        elif reason in ('RESULT_SEALED','RESULT_REJECTED'):
            raw=base64.b64decode(event['payload'],validate=True);result=parsed_result(raw);k=event['activation'];received=event['received_ns']
            if type(received) is not int or event['worker_completed_ns']!=result['completed_ns']:raise AssertionError('Admission timestamp schema')
            takes=[v for v in transports if v['direction']=='plant-results' and v['operation']=='poll' and v['status']=='TAKEN' and v['payload_sha256']==digest(raw) and v['version']==event['slot_version']]
            if len(takes)!=1 or received<takes[0]['end_ns']:raise AssertionError('Admission precedes plant copy/validation')
            predicted=admission_reason(result,takes[0]['key'],issued_seen,published_seen,set(sealed),seen_ids,binding,latched,received,epoch,contract['joint_limits'])
            if predicted!=event['rejection'] or (reason=='RESULT_SEALED')!=(predicted is None):raise AssertionError('Recorded admission decision differs')
            if k!=result['identity']['activation']:raise AssertionError('Result activation mismatch')
            result_hashes.append(digest(raw))
            if predicted is None:sealed[k]=(result['command'],received);seen_ids.add(result['command']['command_id'])
            else:rejections.append({'activation':k,'reason':predicted})
        elif reason=='RESULT_MALFORMED':
            raw=base64.b64decode(event['payload'],validate=True);result_hashes.append(digest(raw))
            malformed=False
            try:
                candidate=parsed_result(raw)
                takes=[v for v in transports if v['direction']=='plant-results' and v['status']=='TAKEN' and v['payload_sha256']==digest(raw)]
                if len(takes)!=1:raise AssertionError('Malformed result missing its transport receipt')
                admission_reason(candidate,takes[0]['key'],issued_seen,published_seen,set(sealed),seen_ids,binding,latched,takes[0]['end_ns'],epoch,contract['joint_limits'])
            except (KeyError,ValueError,TypeError,OverflowError):malformed=True
            if not malformed:raise AssertionError('Valid protocol result mislabeled malformed')
        elif reason!='MAILBOX_BUSY':raise AssertionError('Unknown foundation event reason')
    if set(publication_events)!=set(identities) and metadata.get('event_ledger_overflow') is None and metadata['summary']['foundation']['failure'] is None and terminal_interruption is None:
        raise AssertionError('Unexplained missing publication evidence')
    expected_committed_success=set(published_list)-set(publication_audit['successful_publications_without_committed_event'])
    if published_seen!=expected_committed_success:raise AssertionError('Published set lacks matching committed/reserved status')
    if held_events!=np.flatnonzero(a['control_held']).tolist() and metadata.get('event_ledger_overflow') is None:raise AssertionError('Held event coverage')
    for i in range(1,c):
        if not a['control_held'][i]:
            if i not in sealed:raise AssertionError('Activated command lacks timely sealed result')
            command,admitted=sealed[i]
            if int(a['control_admitted_ns'][i])!=admitted:raise AssertionError('Actual admission does not match sealed timestamp')
            if command['target']!=b64(a['control_target'][i].tobytes()) or command['raw_action']!=b64(a['control_raw_action'][i].tobytes()):raise AssertionError('Activated result bytes differ')
        elif int(a['control_admitted_ns'][i])!=int(a['control_admitted_ns'][i-1]):raise AssertionError('Held admission time changed')
    mismatches=[];worker_hashes=[]
    if worker is not None:
        if worker['recorded_command_table_sha256']!=schedule or worker['native_steps']!=0 or worker['model_calls']!=0:raise AssertionError('Worker table/scope changed')
        if any(type(worker[k]) is not int or worker[k]<0 for k in ('pid','polls','start_ns','end_ns')) or worker['end_ns']<worker['start_ns'] or worker['polls']>60000:
            raise AssertionError('Worker lifecycle clock/counter schema')
        for row in worker['events']:
            if 'operation' in row and (type(row['start_ns']) is not int or type(row['end_ns']) is not int or row['end_ns']<row['start_ns']):
                raise AssertionError('Worker transport time order')
            if row.get('reason')!='RECORDED_REPLY':continue
            k=row['activation'];identity=identities[k]
            if row['job_sha256']!=digest(jobs[str(k)]) or not identity['created_ns']<=row['received_ns']<=row['completed_ns']<=row['published_return_ns']:
                raise AssertionError('Worker exact job/timestamp binding')
            raw=encoded(dict(identity=identity,command=commands[k],completed_ns=row['completed_ns']))
            takes=[v for v in worker['events'] if v.get('direction')=='worker-jobs' and v.get('operation')=='poll' and v.get('status')=='TAKEN' and v.get('key')==k and v.get('payload_sha256')==row['job_sha256']]
            publishes=[v for v in worker['events'] if v.get('direction')=='worker-results' and v.get('operation')=='publish' and v.get('key')==k and v.get('payload_sha256')==row['result_sha256']]
            if len(takes)!=1 or takes[0]['end_ns']>row['received_ns'] or len(publishes)!=1 or publishes[0]['status']!=row['status'] or publishes[0]['start_ns']<row['completed_ns'] or publishes[0]['end_ns']>row['published_return_ns']:
                raise AssertionError('Worker copied/published event identity differs')
            if row['result_sha256']!=digest(raw) or row['command_id']!=commands[k]['command_id']:mismatches.append(k)
            if row['status']=='PUBLISHED':worker_hashes.append(row['result_sha256'])
        if not set(result_hashes)<=set(worker_hashes):raise AssertionError('Plant result lacks successful worker publication')
    elif result_hashes:raise AssertionError('Received results lack worker ledger')
    return dict(issued_jobs=len(identities),published_jobs=len(published_list),admitted_results=len(sealed),
        publication_attempts=publication_audit,rejected_results=rejections,held_controls=held_events,worker_recorded_reply_mismatches=mismatches,
        worker_reply_bytes_exact=not mismatches,recorded_schedule_sha256=schedule,
        complete_protocol_coverage=publication_audit['full_publication_coverage'] and len(identities)==1818 and len(published_list)==1818 and len(sealed)==1818 and not rejections and not held_events and not mismatches)
