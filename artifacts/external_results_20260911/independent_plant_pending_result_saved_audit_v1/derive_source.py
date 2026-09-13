"""Fresh saved-auditor source namespace; never prepare an actual run/request."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;S=B/'source_draft_v1'
O=B.parent/'independent_plant_pending_publication_saved_audit_v1/source_draft_v2'
P=B.parent/'independent_plant_pending_result_v1/source_draft_v1'
S.mkdir(exist_ok=False)
prior={}
for p in sorted(O.glob('*.py')):
    data=p.read_bytes();(S/p.name).write_bytes(data);prior[p.name]=hashlib.sha256(data).hexdigest()
with (B/'original_source_hashes.json').open('x') as f:json.dump(prior,f,indent=2)
def change(name,old,new):
    p=S/name;t=p.read_text();assert t.count(old)==1,(name,old[:90]);p.write_text(t.replace(old,new),newline='\n')
change('clock_protocol_math.py','from retry_publication_math import check as check_publications',
 'from retry_publication_math import check as check_publications\nfrom worker_result_math import check as check_worker')
change('clock_protocol_math.py','schedule_digest=digest(encoded(body[\'active_command\'])),created_ns=created)',
 'schedule_digest=digest(encoded(body[\'active_command\'])),created_ns=created,\n        deadline_ns=metadata[\'summary\'][\'foundation\'][\'epoch_ns\']+activation*20000000)')
change('clock_protocol_math.py',"    if encoded(identity)!=encoded(issued[activation]):return 'JOB_OR_INPUT_MISMATCH'",
 "    if type(identity.get('deadline_ns')) is not int or identity['deadline_ns']!=epoch+activation*20000000:return 'WRONG_ORIGINAL_DEADLINE'\n    if encoded(identity)!=encoded(issued[activation]):return 'JOB_OR_INPUT_MISMATCH'")
p=S/'clock_protocol_math.py';text=p.read_text();start=text.index('    mismatches=[];worker_hashes=[]');end=text.index('    return dict(issued_jobs=',start)
text=text[:start]+'''    worker_audit=check_worker(worker,identities,jobs,commands,epoch,schedule,result_hashes)
    mismatches=worker_audit['recorded_reply_mismatches']
'''+text[end:]
text=text.replace("        worker_reply_bytes_exact=not mismatches,recorded_schedule_sha256=schedule,", "        worker_reply_bytes_exact=not mismatches,worker_result_retry=worker_audit,recorded_schedule_sha256=schedule,")
text=text.replace("and not rejections and not held_events and not mismatches)","and not rejections and not held_events and not mismatches and worker_audit['full_worker_coverage'])")
p.write_text(text,newline='\n')
stage=S/'clock_stage_math.py';text=stage.read_text()
for role,name in [('clock_runner','run_clock.py'),('clock_core','clock_core.py')]:
    import re
    text,n=re.subn(r"('"+role+r"':')[0-9a-f]{64}(')",lambda m:m[1]+hashlib.sha256((P/name).read_bytes()).hexdigest()+m[2],text)
    assert n==1
extra=''.join("    '"+role+"':'"+hashlib.sha256((P/name).read_bytes()).hexdigest()+"',\n" for role,name in [('clock_worker','dummy_worker.py'),('clock_protocol','recorded_protocol.py'),('clock_pending_result','pending_result.py')])
text=text.replace("    'original_core':",extra+"    'original_core':")
text=text.replace("('__init__','_boundary','tick','summary')","('__init__','_boundary','tick','summary','_admit')")
text=text.replace("    expected={'epoch_lead_ns':", "    if request.get('worker_result_retry_contract')!='immutable_result_BUSY_max20_original_deadline' or request.get('job_deadline_contract')!='plant_epoch_plus_activation_20ms':\n        raise AssertionError('Explicit worker retry and original Job deadline contracts required')\n    expected={'epoch_lead_ns':")
stage.write_text(text,newline='\n')
change('prepare_request.py',"    roles['clock_core']=local(run['source_directory']).resolve()/'clock_core.py'", 
 "    roles['clock_core']=local(run['source_directory']).resolve()/'clock_core.py'\n    roles['clock_worker']=local(run['source_directory']).resolve()/'dummy_worker.py'\n    roles['clock_protocol']=local(run['source_directory']).resolve()/'recorded_protocol.py'\n    roles['clock_pending_result']=local(run['source_directory']).resolve()/'pending_result.py'")
change('audit_clock.py',"    args=receipt['exact_wsl_arguments']", 
 "    for key,name in [('clock_worker','dummy_worker.py'),('clock_protocol','recorded_protocol.py'),('clock_pending_result','pending_result.py')]:\n        if roles[key]!=source_directory/name:raise AssertionError('Worker retry source namespace differs')\n    args=receipt['exact_wsl_arguments']")
print(json.dumps({'copied_prior_sources':len(prior),'source_only':True}))
