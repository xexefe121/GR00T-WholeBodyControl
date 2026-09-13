"""Literal request/clearance and role membership checks before task initialization."""
import hashlib
import json
from pathlib import Path
from recovery_contract import protocol,SOURCE_TRACE_SHA,SEMANTICS_SHA,SEMANTICS_OWNER_SHA


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def path(value):
    value=str(value)
    if len(value)>2 and value[1]==':' and Path('/mnt').exists():
        value='/mnt/'+value[0].lower()+value[2:].replace('\\','/')
    return Path(value)


def admit(base):
    request_path=base/'execution_request.json';frozen_path=base/'frozen_inputs.json'
    request=read(request_path);receipt=read(frozen_path);clearance=read(base/'execution_clearance.json')
    if request.get('root_selected') is not True or request.get('protocol')!=protocol():
        raise ValueError('Fixed recovery protocol is not selected')
    if clearance.get('approved') is not True or clearance.get('request_sha256')!=sha(request_path) or clearance.get('frozen_receipt_sha256')!=sha(frozen_path):
        raise ValueError('Wrong concrete recovery clearance')
    review_subject=clearance['review']
    if sha(path(review_subject['path']))!=review_subject['sha256']:raise ValueError('Changed concrete review')
    review=read(path(review_subject['path']))
    if review.get(review_subject['pass_field']) is not True or review.get('request_sha256')!=sha(request_path) or review.get('frozen_receipt_sha256')!=sha(frozen_path):
        raise ValueError('Concrete review does not qualify this request and receipt')
    for relative,digest in receipt['source_sha256'].items():
        if sha(base/'source_snapshot_v1'/relative)!=digest:raise ValueError('Changed frozen source: '+relative)
    for name,digest in receipt['input_sha256'].items():
        if sha(path(name))!=digest:raise ValueError('Changed frozen input: '+name)
    for name,subject in request['subjects'].items():
        if receipt['input_sha256'].get(subject['path'])!=subject['sha256'] or sha(path(subject['path']))!=subject['sha256']:
            raise ValueError('Consumed role absent from frozen inputs: '+name)
    for name,relative in [('selected_snapshot','inputs/precontrol251.npz'),
                          ('selected_prefix','inputs/actual_prefix251.npz'),
                          ('input_selection','inputs/selection_receipt.json')]:
        if path(request['subjects'][name]['path']).resolve()!=(base/relative).resolve():
            raise ValueError('Consumed role path does not match actual setup: '+name)
    for name,expected in [('trace',SOURCE_TRACE_SHA),('semantics',SEMANTICS_SHA),('semantics_owner',SEMANTICS_OWNER_SHA)]:
        if request['subjects'][name]['sha256']!=expected:raise ValueError('Wrong prespecified departure subject: '+name)
    for name in ('nominal','initial_seed','post_lifecycle_hold_5s','initial_restore_preflight.json','ATTEMPT_STARTED','outcome.json','failure.json','work_counters.json'):
        if (base/name).exists():raise ValueError('Previous selected attempt exists: '+name)
    return request
