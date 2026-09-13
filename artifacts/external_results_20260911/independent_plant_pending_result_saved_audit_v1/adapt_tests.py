"""Adapt prior synthetic fixture to explicit deadline and worker retry schema."""
from pathlib import Path
B=Path(__file__).resolve().parent/'source_draft_v1'
p=B/'test_clock_protocol_math.py';s=p.read_text()
s=s.replace('from clock_control_math import SIZES,terms','from clock_control_math import SIZES,terms\nfrom test_worker_result_math import one_case')
where=s.index('\ndef small_case():')
s=s[:where]+'''
def make_worker(identity,job,command,schedule,epoch,shift=0):
    worker=one_case(('PUBLISHED',))[0]
    old_job=worker['events'][1]['job_sha256'];old_result=worker['events'][1]['result_sha256']
    raw=encoded(dict(identity=identity,command=command,completed_ns=epoch+shift+100))
    for event in worker['events']:
        if event.get('direction')=='worker-results':event.update(start_ns=epoch+101,end_ns=epoch+105)
        if event.get('reason')=='WORKER_RESULT_PUBLICATION':event.update(start_ns=epoch+100,end_ns=epoch+106)
        if event.get('reason')=='WORKER_RESULT_READY':event.update(identity=identity,payload=b64(raw),payload_ready_ns=epoch+100)
        if event.get('reason')=='WORKER_RESULT_TERMINAL':event.update(observed_ns=epoch+106)
        for key in ('start_ns','end_ns','received_ns','completed_ns','payload_ready_ns','observed_ns'):
            if key in event:event[key]+=shift
        for key in ('payload_sha256','job_sha256','result_sha256'):
            if event.get(key)==old_job:event[key]=digest(job)
            elif event.get(key)==old_result:event[key]=digest(raw)
    d=worker['result_retry']['last_result']
    d.update(source=dict(slot=1,key=1,version=1,payload=b64(job),payload_sha256=digest(job)),identity=identity,
        received_ns=epoch+shift+30,completed_ns=epoch+shift+100,payload=b64(raw),payload_sha256=digest(raw),
        payload_ready_ns=epoch+shift+100,attempt_start_ns=epoch+shift+100,attempt_end_ns=epoch+shift+106,
        terminal_ns=epoch+shift+106)
    worker.update(recorded_command_table_sha256=schedule,start_ns=epoch-1000,end_ns=epoch+shift+1000)
    return worker

''' + s[where:]
s=s.replace("schedule_digest=digest(encoded(commands[0])),created_ns=epoch+1)","schedule_digest=digest(encoded(commands[0])),created_ns=epoch+1,deadline_ns=epoch+20000000)")
start=s.index('    worker=dict(recorded_command_table_sha256=');end=s.index('\n    request=dict(',start)
s=s[:start]+"    worker=make_worker(identity,job,commands[1],schedule,epoch)"+s[end:]
start=s.index("        worker=args[4];worker['end_ns']+=shift");end=s.index('        r=protocol(*args)',start)
s=s[:start]+"        args[4]=make_worker(result['identity'],args[2]['1'],result['command'],args[5]['command_table_sha256'],epoch,shift)\n"+s[end:]
s=s.replace("args[4]['events'][-1]['received_ns']=0","args[4]['events'][1]['received_ns']=0")
start=s.index('    def test_worker_output_mismatch_is_command_failure(self):');end=s.index('    def test_worker_publication_must_match_taken_bytes(self):',start)
s=s[:start]+'''    def test_worker_output_mismatch_is_command_failure(self):
        args=list(small_case());worker=args[4];ready=worker['events'][1]
        raw=json.loads(__import__('base64').b64decode(ready['payload']));raw['command']['command_id']='wrong'
        payload=encoded(raw);old=ready['result_sha256'];new=digest(payload)
        for event in worker['events']:
            for key in ('payload_sha256','result_sha256'):
                if event.get(key)==old:event[key]=new
            if event.get('reason')=='WORKER_RESULT_READY':event['payload']=b64(payload)
        worker['result_retry']['last_result'].update(payload=b64(payload),payload_sha256=new)
        event=json.loads(args[1]['foundation_events'][-1]);event['payload']=b64(payload);args[1]['foundation_events'][-1]=encoded(event)
        event=json.loads(args[1]['transport_records'][-1]);event['payload_sha256']=new;args[1]['transport_records'][-1]=encoded(event)
        r=protocol(*args);self.assertEqual(r['worker_recorded_reply_mismatches'],[1]);self.assertFalse(r['complete_protocol_coverage'])
''' + s[end:]
s=s.replace("args[4]['events'][1]['payload_sha256']='0'*64","args[4]['events'][2]['payload_sha256']='0'*64")
s=s.replace("self.assertRaisesRegex(AssertionError,'copied/published')","self.assertRaisesRegex(AssertionError,'transport')")
p.write_text(s,newline='\n')
p=B/'test_clock_stage_math.py';s=p.read_text()
s=s.replace("request={'job_publication_retry_contract':", "request={'worker_result_retry_contract':'immutable_result_BUSY_max20_original_deadline','job_deadline_contract':'plant_epoch_plus_activation_20ms','job_publication_retry_contract':")
s=s.replace("'clock_runner':new/'independent_plant_clock_timeout_correction_v1/source_draft_v1/run_clock.py'", "'clock_runner':new/'independent_plant_pending_result_v1/source_draft_v1/run_clock.py'")
s=s.replace("'clock_core':new/'independent_plant_pending_publication_v1/source_draft_v1/clock_core.py'", "'clock_core':new/'independent_plant_pending_result_v1/source_draft_v1/clock_core.py',\n        'clock_worker':new/'independent_plant_pending_result_v1/source_draft_v1/dummy_worker.py',\n        'clock_protocol':new/'independent_plant_pending_result_v1/source_draft_v1/recorded_protocol.py',\n        'clock_pending_result':new/'independent_plant_pending_result_v1/source_draft_v1/pending_result.py'")
p.write_text(s,newline='\n')
