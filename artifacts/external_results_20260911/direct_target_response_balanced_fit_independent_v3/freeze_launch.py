"""Freeze the actual reviewed invocation. No audit or task model execution."""
import argparse,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main(args):
    request_path=BASE/'audit_request.json';assert sha(request_path)==args.request_sha256
    request=read(request_path);assert request['kind']=='saved_response_balanced_warm_only'
    owner=request['subjects']['owner_completion'];assert sha(owner['path'])==owner['sha256']
    review=read(request['source_review']['path']);assert sha(request['source_review']['path'])==request['source_review']['sha256'] and review[request['source_review']['pass_field']] is True
    for name in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py'):assert review['helper_sha256'][name]==sha(BASE/name)
    pins=dict(request['source_sha256']);pins.update({s['path']:s['sha256'] for s in request['subjects'].values()})
    pins[request['source_review']['path']]=request['source_review']['sha256']
    for p in [request_path,BASE/'run_audit_durable.ps1',BASE/'prepare_audit_request.py',BASE/'source_preparation.json',BASE/'verify_completion.py',Path(request['python_path']),Path(__file__)]:pins[p.resolve().as_posix()]=sha(p)
    for p,d in pins.items():assert sha(p)==d
    receipt=dict(selected_single_saved_audit=True,kind='saved_response_balanced_warm_only',request_sha256=args.request_sha256,
        source_review_sha256=request['source_review']['sha256'],python_path=request['python_path'],input_sha256=pins,
        fit_owner_sha256=owner['sha256'],automatic_retry=False,task_model_calls=0,native_steps=0)
    with (BASE/'launch_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(launch_receipt_sha256=sha(BASE/'launch_receipt.json'),pins=len(pins))))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request-sha256',required=True);main(p.parse_args())
