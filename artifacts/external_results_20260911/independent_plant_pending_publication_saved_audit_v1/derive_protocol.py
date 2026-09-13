"""One source-only adaptation of the completed old saved audit; no task inputs."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_draft_v1'
def change(text,before,after):
    assert text.count(before)==1,(before[:70],text.count(before))
    return text.replace(before,after)
def edit(name,replacements):
    p=SOURCE/name;text=p.read_text()
    for before,after in replacements:text=change(text,before,after)
    p.write_text(text,encoding='utf-8',newline='\n')

edit('clock_protocol_math.py',[
 ('from clock_saved_math import exact','from clock_saved_math import exact\nfrom retry_publication_math import check as check_publications'),
 ("    issued_seen={};published_seen=set();sealed={};seen_ids={commands[0]['command_id']};latched=False", """    publication_audit=check_publications(identities,jobs,events,transports,metadata['summary']['foundation'],epoch,a,published_list,
        metadata.get('event_ledger_overflow'),metadata.get('transport_ledger_overflow'))
    issued_seen={};published_seen=set();sealed={};seen_ids={commands[0]['command_id']};latched=False"""),
 ("""            if event['input_digest']!=identity['input_digest'] or event['source_window']!=identity['window_id'] or physics!=identity['snapshot_physics']:
                raise AssertionError('Job publication does not match exact issued identity')
            matches=[v for v in transports if v['direction']=='plant-jobs' and v['operation']=='publish' and v['key']==k and v['payload_sha256']==digest(jobs[str(k)])]
            if len(matches)!=1 or matches[0]['status']!=event['status']:raise AssertionError('Job mailbox/publication ledger differs')""", """            # Every retry identity, tick, guard, transport status and counter was
            # independently checked above. Admission still follows event order."""),
 ("        elif reason=='HELD_COMMAND_INTERVAL':", "        elif reason=='PENDING_JOB_EXPIRED':\n            pass  # Exact BUSY state, fixed deadline and counters checked above.\n        elif reason=='HELD_COMMAND_INTERVAL':"),
 ("    if len(publication_events)!=len(set(publication_events)):raise AssertionError('Duplicate job publication event')\n", ""),
 ("        rejected_results=rejections,held_controls=held_events,worker_recorded_reply_mismatches=mismatches,", "        publication_attempts=publication_audit,rejected_results=rejections,held_controls=held_events,worker_recorded_reply_mismatches=mismatches,"),
 ("complete_protocol_coverage=len(identities)==1818", "complete_protocol_coverage=publication_audit['full_publication_coverage'] and len(identities)==1818")])

edit('clock_stage_math.py',[
 ("'clock_watchdog':'a6613da97b1163031b6b7641af62f4c9ec63662bc3c6ff25deda3ee09b0b8089'}", """'clock_watchdog':'a6613da97b1163031b6b7641af62f4c9ec63662bc3c6ff25deda3ee09b0b8089',
    'clock_core':'c67d970962a1fe67258154d13beb82c04a69f70c25426e77736def8c790770d1',
    'original_core':'9bc6f725c469ded387b15d1e829ba0c16ed6692baab8b7d00dbe66fb3e3a2432'}"""),
 ("    return {'literal_sources_exact':True,'original_hot_loop_AST_exact':True,'producer_imports':0}", """    def methods(text):
        node=next(v for v in ast.parse(text).body if isinstance(v,ast.ClassDef) and v.name=='PlantFoundation')
        return {v.name:ast.dump(v,include_attributes=False) for v in node.body if isinstance(v,ast.FunctionDef)}
    old,new=methods(texts['original_core']),methods(texts['clock_core'])
    unchanged=[k for k in old if k not in ('__init__','_boundary','tick','summary')]
    if any(old[k]!=new[k] for k in unchanged):raise AssertionError('Unchanged admission/clock methods differ')
    return {'literal_sources_exact':True,'fixed_outer_loop_AST_exact':True,
            'foundation_tick_changed_for_bounded_retry':True,'unchanged_foundation_methods':unchanged,'producer_imports':0}"""),
 ("def request_contract(request,arguments):\n", """def request_contract(request,arguments):
    if request.get('job_publication_retry_contract')!='pending_BUSY_same_job_max10_before_original_activation':
        raise AssertionError('New retry source requires explicit matching request contract')
""")])

edit('prepare_request.py',[
 ("    roles['clock_watchdog']=local(run['source_directory']).resolve()/'stage_watchdog.py'", """    roles['clock_watchdog']=local(run['source_directory']).resolve()/'stage_watchdog.py'
    roles['clock_core']=local(run['source_directory']).resolve()/'clock_core.py'
    roles['original_core']=source.parents[1]/'independent_plant_clock_timeout_correction_v1/source_draft_v1/clock_core.py'""")])
edit('audit_clock.py',[
 ("    if roles['clock_runner']!=source_directory/'run_clock.py' or roles['clock_watchdog']!=source_directory/'stage_watchdog.py':raise AssertionError('Actual clock source namespace differs')", "    if roles['clock_runner']!=source_directory/'run_clock.py' or roles['clock_watchdog']!=source_directory/'stage_watchdog.py' or roles['clock_core']!=source_directory/'clock_core.py':raise AssertionError('Actual clock source namespace differs')")])
