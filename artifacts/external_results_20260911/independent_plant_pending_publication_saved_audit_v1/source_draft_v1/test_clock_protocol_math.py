"""Synthetic bytes and timestamps only; no producer, model, worker or native calls."""
import copy,json,unittest
import numpy as np
from clock_protocol_math import encoded,b64,digest,command_wire,canonical_commands,job_expected,admission_reason,parsed_result,protocol
from clock_control_math import SIZES,terms


def small_case():
    full=dict(target=np.zeros((1569,23)),action=np.zeros((1569,23),np.float32))
    hold=dict(target=np.zeros((250,23)),action=np.zeros((250,23),np.float32))
    full['action'][1,0]=-0.0
    commands,frames,schedule,windows=canonical_commands(full,hold,'a'*64,'b'*64)
    epoch=100_000_000;binding=dict(run='synthetic',model_hash='c'*64,reference_hash='d'*64,epoch=1)
    state=np.zeros(291);state[3]=.75;state[4]=1.
    h={k:np.zeros((4,v),np.float32) for k,v in SIZES.items()};incoming=np.zeros(23,np.float32)
    measured=terms(state,incoming,np.zeros(23));before=[(k,h[k].tobytes()) for k in sorted(h)]
    after=[(k,np.concatenate([measured[k][None],h[k][:3]]).tobytes()) for k in sorted(h)]
    named=[(k,measured[k].tobytes()) for k in sorted(h)]
    a=dict(control_control=np.array([0],np.int64),control_actual_activation_ns=np.array([epoch],np.int64),
        control_admitted_ns=np.array([epoch-100],np.int64),control_integration=state[None],control_incoming_raw=incoming[None],
        control_target=full['target'][:1].copy(),control_raw_action=full['action'][:1].copy(),control_flat_history_before=np.zeros((1,300),np.float32),
        control_held=np.zeros(1,bool),control_nominal_source_frame=np.array([11],np.int64),control_active_source_frame=np.array([11],np.int64))
    metadata=dict(control_command_ids=[commands[0]['command_id']],control_nominal_windows=[windows[0]],control_active_windows=[windows[0]],
        control_history_before=[before],control_history_after=[after],control_measured_terms=[named],summary=dict(foundation=dict(epoch_ns=epoch,failure=None)))
    body=dict(state=b64(state.tobytes()),incoming_raw=b64(incoming.tobytes()),active_command=commands[0],
        history_before=[(k,b64(v)) for k,v in before],terms=[(k,b64(v)) for k,v in named],history_after=[(k,b64(v)) for k,v in after])
    payload=encoded(body)
    identity=dict(binding=binding,sequence=0,snapshot_control=0,snapshot_physics=0,activation=1,window_id=windows[1],
        input_digest=digest(payload),history_digest=digest(a['control_flat_history_before'][0].tobytes()),schedule_digest=digest(encoded(commands[0])),created_ns=epoch+1)
    job=encoded(dict(identity,snapshot_payload=b64(payload)))
    a['step_actual_start']=np.array([epoch],np.int64)
    a['step_actual_end']=np.array([epoch+1000],np.int64)
    metadata['summary']['foundation'].update(job_publication_attempted=1,job_publication_returned=1,
        job_publication_events=1,pending_expirations=0,pending_publication=None,
        last_job_publication=dict(identity=identity,payload=b64(job),deadline_ns=epoch+20000000,
            attempts=1,last_attempt_index=0,last_status='PUBLISHED',event_recorded=True,retry_checked_ns=None))
    result=encoded(dict(identity=identity,command=commands[1],completed_ns=epoch+100))
    takes=dict(direction='plant-results',operation='poll',status='TAKEN',key=1,version=1,payload_sha256=digest(result),start_ns=epoch+110,end_ns=epoch+120)
    publication=dict(direction='plant-jobs',operation='publish',status='PUBLISHED',key=1,version=1,payload_sha256=digest(job),start_ns=epoch+2,end_ns=epoch+3)
    metadata['transport_records']=[encoded(publication),encoded(takes)]
    metadata['foundation_events']=[encoded(dict(physics=0,reason='JOB_PUBLICATION',activation=1,input_digest=identity['input_digest'],source_window=windows[1],status='PUBLISHED',attempt=1,deadline_ns=epoch+20000000,retry_checked_ns=None,payload_sha256=digest(job))),
        encoded(dict(physics=0,reason='RESULT_SEALED',rejection=None,activation=1,received_ns=epoch+130,worker_completed_ns=epoch+100,slot_version=1,payload=b64(result)))]
    worker=dict(recorded_command_table_sha256=schedule,native_steps=0,model_calls=0,pid=3,polls=1,start_ns=epoch-1000,end_ns=epoch+1000,
        events=[dict(direction='worker-jobs',operation='poll',status='TAKEN',key=1,payload_sha256=digest(job),start_ns=epoch+10,end_ns=epoch+20),
                dict(direction='worker-results',operation='publish',status='PUBLISHED',key=1,payload_sha256=digest(result),start_ns=epoch+101,end_ns=epoch+105),
                dict(reason='RECORDED_REPLY',activation=1,received_ns=epoch+30,completed_ns=epoch+100,published_return_ns=epoch+106,
                    status='PUBLISHED',job_sha256=digest(job),result_sha256=digest(result),command_id=commands[1]['command_id'])])
    request=dict(command_table_sha256=schedule,run_id='synthetic',expected_model_sha256='c'*64,input_epoch=1)
    return a,metadata,{'1':job},[1],worker,request,full,hold,'a'*64,'b'*64,'d'*64,dict(joint_limits=np.tile([-1.,1.],(23,1)))


class ProtocolTests(unittest.TestCase):
    def test_retry_then_sealed_uses_original_identity(self):
        args=list(small_case());a,metadata=args[:2];epoch=100000000;shift=2000000
        publication=json.loads(metadata['foundation_events'][0]);publication['status']='BUSY'
        retry=dict(publication,status='PUBLISHED',physics=1,attempt=2,retry_checked_ns=epoch+shift+1)
        result_event=json.loads(metadata['foundation_events'][1]);result=json.loads(__import__('base64').b64decode(result_event['payload']))
        result['completed_ns']+=shift;raw=encoded(result)
        result_event.update(physics=2,payload=b64(raw),received_ns=epoch+shift+130,worker_completed_ns=epoch+shift+100)
        metadata['foundation_events']=[encoded(publication),encoded(retry),encoded(result_event)]
        oldpub=json.loads(metadata['transport_records'][0]);oldpub['status']='BUSY'
        newpub=dict(oldpub,status='PUBLISHED',start_ns=epoch+shift+2,end_ns=epoch+shift+3)
        take=json.loads(metadata['transport_records'][1]);take.update(payload_sha256=digest(raw),start_ns=epoch+shift+110,end_ns=epoch+shift+120)
        metadata['transport_records']=[encoded(oldpub),encoded(newpub),encoded(take)]
        f=metadata['summary']['foundation'];f.update(job_publication_attempted=2,job_publication_returned=2,job_publication_events=2)
        f['last_job_publication'].update(attempts=2,last_attempt_index=1,retry_checked_ns=epoch+shift+1)
        a['step_actual_start']=np.array([epoch,epoch+shift,epoch+4000000],np.int64)
        a['step_actual_end']=a['step_actual_start']+50
        worker=args[4];worker['end_ns']+=shift
        for event in worker['events']:
            for key in ('start_ns','end_ns','received_ns','completed_ns','published_return_ns'):
                if key in event:event[key]+=shift
            if 'result_sha256' in event:event['result_sha256']=digest(raw)
            if event.get('direction')=='worker-results':event['payload_sha256']=digest(raw)
        r=protocol(*args)
        self.assertEqual(r['admitted_results'],1)
        self.assertEqual(r['publication_attempts']['retry_attempts'],1)
        self.assertFalse(r['complete_protocol_coverage'])

    def test_complete_partial_protocol_no_full_credit(self):
        r=protocol(*small_case());self.assertEqual(r['admitted_results'],1);self.assertFalse(r['complete_protocol_coverage'])
    def test_actual_history_state_job_bytes_bound(self):
        for key,index in [('control_integration',3),('control_incoming_raw',2),('control_flat_history_before',2)]:
            args=list(small_case());args[0][key][0,index]=.1
            with self.assertRaisesRegex(AssertionError,'canonical job'):protocol(*args)
    def test_source_window_mutation_rejected(self):
        args=list(small_case());args[1]['control_nominal_windows'][0]='wrong'
        with self.assertRaisesRegex(AssertionError,'source windows'):protocol(*args)
    def test_admission_is_after_copy(self):
        args=list(small_case());event=__import__('json').loads(args[1]['foundation_events'][-1]);event['received_ns']=100000119
        args[1]['foundation_events'][-1]=encoded(event)
        with self.assertRaisesRegex(AssertionError,'precedes plant'):protocol(*args)
    def test_worker_bad_job_timestamp_rejected(self):
        args=list(small_case());args[4]['events'][-1]['received_ns']=0
        with self.assertRaisesRegex(AssertionError,'job/timestamp'):protocol(*args)
    def test_worker_output_mismatch_is_command_failure(self):
        args=list(small_case());args[4]['events'][-1]['command_id']='wrong'
        r=protocol(*args);self.assertEqual(r['worker_recorded_reply_mismatches'],[1]);self.assertFalse(r['complete_protocol_coverage'])
    def test_worker_publication_must_match_taken_bytes(self):
        args=list(small_case());args[4]['events'][1]['payload_sha256']='0'*64
        with self.assertRaisesRegex(AssertionError,'copied/published'):protocol(*args)
    def test_initial_admission_cannot_move_to_epoch(self):
        args=list(small_case());args[0]['control_admitted_ns'][0]=100000000
        with self.assertRaisesRegex(AssertionError,'before the fixed'):protocol(*args)
    def test_rejection_order_and_deadlines(self):
        args=small_case();job=__import__('json').loads(args[2]['1']);identity={k:v for k,v in job.items() if k!='snapshot_payload'}
        commands=canonical_commands(args[6],args[7],args[8],args[9])[0]
        result=dict(identity=identity,command=commands[1],completed_ns=100000100)
        def reason(value,**kw):
            defaults=dict(key=1,issued={1:identity},published={1},seen_activations=set(),seen_command_ids={commands[0]['command_id']},binding=identity['binding'],held_latched=False,received=100000130,epoch=100000000,limits=args[-1]['joint_limits'])
            defaults.update(kw);return admission_reason(value,**defaults)
        self.assertIsNone(reason(result));self.assertEqual(reason(result,received=120000000),'LATE')
        self.assertEqual(reason(result,key=2),'WRONG_ACTIVATION');self.assertEqual(reason(result,seen_activations={1}),'DUPLICATE')
        self.assertEqual(reason(result,published=set()),'JOB_NOT_PUBLISHED');self.assertEqual(reason(result,issued={}),'UNKNOWN_JOB')
        altered=copy.deepcopy(result);altered['identity']['binding']['epoch']=2
        self.assertEqual(reason(altered),'WRONG_BINDING_OR_EPOCH');self.assertEqual(reason(altered,held_latched=True),'FAULT_LATCHED')
        altered=copy.deepcopy(result);altered['completed_ns']=100000200
        self.assertEqual(reason(altered),'FUTURE_WORKER_TIMESTAMP')
        altered['completed_ns']=100000000;self.assertEqual(reason(altered),'WORKER_TIMESTAMP_BEFORE_JOB')
    def test_identical_targets_different_raw_bytes_remain_distinct(self):
        args=list(small_case());commands=canonical_commands(args[6],args[7],args[8],args[9])[0]
        self.assertEqual(commands[0]['target'],commands[1]['target']);self.assertNotEqual(commands[0]['raw_action'],commands[1]['raw_action'])


if __name__=='__main__':unittest.main()
