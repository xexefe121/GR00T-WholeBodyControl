"""Post-stop saved-output accounting; no model, feature, map or native calls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def verify(absence_path):
    request=read(BASE/'request.json');report=read(BASE/'generation/report.json');manifest=read(BASE/'generation/manifest.json')
    start=read(BASE/'process/start.json');child=read(BASE/'process/child.json');exit_record=read(BASE/'process/exit.json');absence=read(absence_path)
    if exit_record['exit_code']!=0 or exit_record['raw_python_exit_code']!=0 or not exit_record['raw_exit_known'] or exit_record['error'] is not None:raise ValueError('known successful stopped process required')
    if not absence['wrapper_absent'] or not absence['child_absent']:raise ValueError('process still present')
    if absence['wrapper_pid']!=start['wrapper_pid'] or absence['child_pid']!=child['child_pid'] or child['wrapper_pid']!=start['wrapper_pid']:raise ValueError('process identity')
    if exit_record['wrapper_pid']!=start['wrapper_pid'] or exit_record['child_pid']!=child['child_pid']:raise ValueError('exit PID relation')
    request_sha=sha(BASE/'request.json')
    if any(x['request_sha256']!=request_sha for x in (report,manifest,start,exit_record)):raise ValueError('actual request binding')
    review_path=Path(start['review_path'])
    if sha(review_path)!=start['review_sha256'] or exit_record['review_sha256']!=start['review_sha256']:raise ValueError('final review changed')
    for name in ('preflight_hashes.json','postrun_hashes.json'):
        ledger=read(BASE/'process'/name)
        if not ledger['all_exact'] or set(ledger['files'])!=set(request['windows_launch_sha256']):raise ValueError('launch map coverage')
        for path,expected in request['windows_launch_sha256'].items():
            row=ledger['files'][path]
            if row!={'expected':expected,'actual':expected,'matched':True} or sha(Path(path))!=expected:raise ValueError('changed launch pin '+path)
    for name,digest in report['output_sha256'].items():
        if sha(BASE/'generation'/name)!=digest:raise ValueError('changed output '+name)
    if sha(BASE/'generation/manifest.json')!=report['manifest_sha256']:raise ValueError('manifest')
    for filename,key in (('group_cell_summary.json','group_cell_summary_sha256'),('input_schema_gate.json','input_schema_gate_sha256'),('center_gate.json','center_gate_sha256'),('overlap_gate.json','overlap_gate_sha256')):
        if sha(BASE/'generation'/filename)!=manifest[key]:raise ValueError('saved gate '+filename)
    for name,spec in manifest['arrays'].items():
        path=BASE/'generation'/spec['path'];a=np.load(path,mmap_mode='r',allow_pickle=False)
        if list(a.shape)!=spec['shape'] or str(a.dtype)!=spec['dtype'] or sha(path)!=spec['sha256']:raise ValueError('saved schema/hash '+name)
        if name=='status' and not np.all(a==1):raise ValueError('uncommitted requested rows')
    if report['centers']!=3057 or report['signed_rows']!=354612 or report['overlap_rows']!=140622 or report['new_rows']!=213990:raise ValueError('counts')
    if report['call_accounting']!={k:357669 for k in ('feature_attempted','feature_returned','map_attempted','map_returned')}:raise ValueError('pure call counts')
    if not report['all_runtime_and_sources_unchanged'] or not report['all_inputs_unchanged']:raise ValueError('producer rehash gate')
    summary=read(BASE/'generation/group_cell_summary.json')['cells']
    if len(summary)!=54 or sum(r['requested_signed_rows'] for r in summary)!=354612:raise ValueError('54 group/cell coverage')
    result={'passed':True,'request_sha256':request_sha,'report_sha256':sha(BASE/'generation/report.json'),'manifest_sha256':sha(BASE/'generation/manifest.json'),
        'source_sha256':sha(Path(__file__)),'process_exit_sha256':sha(BASE/'process/exit.json'),'process_absence_sha256':sha(absence_path),
        'process_absence':absence,'windows_launch_pins_exact':len(request['windows_launch_sha256']),
        'runtime_rehash_scope':'Consumes producer final runtime/source rehash verdict; does not re-execute or independently rehash Linux runtime here.',
        'counts':{'centers':3057,'signed_rows':354612,'old_overlap':140622,'new_added':213990},
        'model_calls':0,'feature_calls':0,'map_calls':0,'physics_steps':0,'independent_semantic_audit_pending':True}
    out=BASE/'owner_completion_verification.json'
    if out.exists():raise ValueError('preserve existing owner receipt')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'sha256':sha(out)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--process-absence',required=True);a=p.parse_args();verify(Path(a.process_absence))
