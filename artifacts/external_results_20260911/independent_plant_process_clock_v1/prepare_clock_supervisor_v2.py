"""Narrow unlaunched supervisor correction. Native runner/request remain unchanged."""
import argparse,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
FOLDER=BASE/'clock_process'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--final-review',type=Path);p.add_argument('--pass-field',default='prelaunch_review_pass');p.add_argument('--root-selected',action='store_true');a=p.parse_args()
    receipt_path=FOLDER/'launch_receipt_v2.json';request=BASE/'clock_request.json'
    if (FOLDER/'started.lock').exists() or (BASE/'run').exists() or (FOLDER/'launch_clearance.json').exists():raise ValueError('Preserve prior clearance/attempt; never retry')
    if a.final_review:
        if not a.root_selected:raise ValueError('Parent selection required')
        review=read(a.final_review)
        if review.get(a.pass_field) is not True:raise ValueError('Actual review must pass')
        for key,path in [('request_subject',request),('launch_receipt_subject',receipt_path)]:
            if Path(review[key]['path']).resolve()!=path.resolve() or review[key]['sha256']!=sha(path):raise ValueError('Literal review subject differs')
        write(FOLDER/'launch_clearance.json',dict(root_selected_single_run=True,request_sha256=sha(request),launch_receipt_sha256=sha(receipt_path),
            review=dict(path=a.final_review.resolve().as_posix(),sha256=sha(a.final_review),pass_field=a.pass_field),automatic_retry=False,hardware_authorized=False))
        print(json.dumps(dict(clearance_sha256=sha(FOLDER/'launch_clearance.json'),launched=False)));return
    if a.root_selected:raise ValueError('Source freeze alone does not select a run')
    original=FOLDER/'run_durable.ps1';text=original.read_text()
    old=r"$receiptPath.Replace('\\','/')"
    if text.count(old)!=1 or text.count("'launch_receipt.json'")!=1:raise ValueError('Exact original two-token substitution required')
    updated=text.replace(old,'$receiptPath.Replace([char]92,[char]47)').replace("'launch_receipt.json'","'launch_receipt_v2.json'")
    corrected=FOLDER/'run_durable_v2.ps1'
    with corrected.open('x',newline='') as stream:stream.write(updated)
    old_owner=BASE/'verify_completion.py';owner_text=old_owner.read_text()
    if owner_text.count("'launch_receipt.json'")!=1:raise ValueError('Owner literal receipt substitution required')
    new_owner=BASE/'verify_completion_v2.py'
    with new_owner.open('x',newline='') as stream:stream.write(owner_text.replace("'launch_receipt.json'","'launch_receipt_v2.json'"))
    old_receipt=FOLDER/'launch_receipt.json';receipt=read(old_receipt)
    for path,digest in receipt['input_hashes'].items():
        if sha(Path(path))!=digest:raise ValueError('Original frozen input changed: '+path)
    pins=dict(receipt['input_hashes'])
    for path in (old_receipt,corrected,new_owner,Path(__file__)):pins[path.resolve().as_posix()]=sha(path)
    receipt=dict(receipt,input_hashes=pins,preserved_original_launch_receipt_sha256=sha(old_receipt),
        supervisor_path=corrected.as_posix(),owner_verifier_path=new_owner.as_posix(),
        unchanged_native_request_sha256=sha(request),correction='Explicit PowerShell char92 to char47 path normalization; separate literal v2 receipt.')
    write(receipt_path,receipt)
    write(BASE/'supervisor_preparation_v2.json',dict(preparation_only=True,actual_launch=False,native_steps=0,model_calls=0,
        source_sha256={p.name:sha(p) for p in (original,corrected,old_owner,new_owner,Path(__file__))},
        original_request_sha256=sha(request),original_launch_receipt_sha256=sha(old_receipt),launch_receipt_sha256=sha(receipt_path),
        unchanged_wsl_arguments=receipt['exact_wsl_arguments'],substitutions=['PowerShell path normalization','Supervisor literal receipt filename','Owner literal receipt filename']))
    print(json.dumps(dict(launch_receipt_sha256=sha(receipt_path),pins=len(pins),request_sha256=sha(request),launched=False)))
if __name__=='__main__':main()
