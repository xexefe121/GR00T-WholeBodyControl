"""Saved-file completion accounting only; no inference, native or process launch."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','evaluation'],required=True)
    p.add_argument('--process-check',type=Path,required=True);a=p.parse_args()
    base=Path(__file__).resolve().parent;folder=base/(a.mode+'_process')
    process=read(a.process_check);assert process['wrapper_absent'] is True and process['child_absent'] is True
    start=read(folder/'start.json');child=read(folder/'child.json')
    assert process['wrapper_pid']==start['wrapper_pid'] and process['child_pid']==child['child_pid']
    receipt=read(folder/'launch_receipt.json');post=read(folder/'postrun_hashes.json')
    assert post==receipt['input_hashes']
    for path,digest in post.items():assert sha(path)==digest,path
    clearance=read(folder/'launch_clearance.json')
    assert sha(folder/'launch_receipt.json')==start['receipt_sha256']==clearance['launch_receipt_sha256']
    assert sha(folder/'launch_clearance.json')==start['clearance_sha256']
    assert sha(clearance['review']['path'])==clearance['review']['sha256']==start['review_sha256']
    raw=read(folder/'raw_exit.json');exit_record=read(folder/'exit.json');verdict=read(folder/'diagnostic_verdict.json')
    assert raw['raw_python_exit_code']==verdict['raw_python_exit_code']
    assert exit_record['error'] is None and exit_record['all_postrun_hashes_exact'] is True
    assert exit_record['raw_child_exit_code']==exit_record['exit_code']==verdict['diagnostic_exit_code']
    paths=[folder/x for x in ('start.json','child.json','launch_receipt.json','launch_clearance.json','raw_exit.json','exit.json','postrun_hashes.json','diagnostic_verdict.json','stdout.log','stderr.log')]
    paths.extend([a.process_check,Path(__file__)])
    if a.mode=='witness':paths.extend((base/'head_witness').glob('*'))
    else:
        for sub in ('nominal','post_lifecycle_hold_5s'):paths.extend(p for p in (base/sub).rglob('*') if p.is_file())
        paths.extend(p for p in base.glob('*.json') if p.name.startswith(('canonical_','actual_query250','pilot_outcome')))
    report=dict(owner_completion_accounting_passed=True,mode=a.mode,raw_python_exit_code=raw['raw_python_exit_code'],
        diagnostic_exit_code=verdict['diagnostic_exit_code'],diagnostic_passed=verdict['passed'],
        diagnostic_reasons=verdict['reasons'],pins_exact=len(post),process_absence=process,
        output_hashes={x.resolve().as_posix():sha(x) for x in paths},
        independent_physics_and_intent_audit_required=a.mode=='evaluation',behavioral_qualification=False)
    destination=base/(a.mode+'_completion_verification.json')
    with destination.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(destination),sha256=sha(destination),diagnostic_passed=verdict['passed'],pins_exact=len(post))))

if __name__=='__main__':main()
