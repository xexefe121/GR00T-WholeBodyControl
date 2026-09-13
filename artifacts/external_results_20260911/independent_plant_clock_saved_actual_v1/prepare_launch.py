"""Frozen source-audit-v3 launcher metadata only; dispatch is always separate."""
import argparse,json,re,hashlib
from pathlib import Path

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SOURCE=NEW/'independent_plant_clock_saved_root_review_v1/source_audit_v3'
SOURCE_REVIEW=NEW/'independent_plant_clock_saved_root_review_v1/root_source_review_v3.json'
SOURCE_REVIEW_SHA='278c509418ef1c7a1be655693a840217dd8c1894a6a6c33f0757a712ad8b9ac7'
HELPERS=('prepare_launch.py','verify_completion.py','test_launch.py','record_helper_preparation.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def linux(path):
    text=Path(path).resolve().as_posix()
    assert text[1]==':' and not any(c.isspace() for c in text)
    return '/mnt/'+text[0].lower()+text[2:]


def replace(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def render(request_sha):
    assert re.fullmatch('[0-9a-f]{64}',request_sha)
    text=(BASE/'preserved_saved_audit_template.ps1.txt').read_text()
    text=replace(text,'8694708a3370fdd26e7f6f4dccb321a7c5afa1ba381a7f9df670511d953e6dcf',request_sha)
    old='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_context_saved_semantics_review_v1'
    text=replace(text,'PYTHONPATH='+old,'PYTHONPATH='+linux(SOURCE))
    text=replace(text,old+'/audit_saved.py',linux(SOURCE/'audit_clock.py')+' --request '+linux(BASE/'request.json')+' --output '+linux(BASE/'results_v1'))
    old_check="if($result.evidence_audit_passed -ne $true -or $result.request_sha256 -ne $receipt.request_sha256 -or $result.context_condition -ne 'causal' -or $result.features -ne 1323 -or $result.model_calls -ne 0 -or $result.native_steps -ne 0){throw 'Saved evidence audit did not pass exact request/scope.'}"
    new_check="if($result.evidence_integrity_passed -ne $true -or $result.request_sha256 -ne $receipt.request_sha256 -or $result.model_calls -ne 0 -or $result.native_steps_executed -ne 0 -or $result.optimizer_updates -ne 0){throw 'Saved clock evidence integrity did not pass exact request/scope.'}"
    return replace(text,old_check,new_check)


def fresh():
    for name in ('process_v1','results_v1','dispatch.json','launch_clearance.json','owner_completion.json'):
        assert not (BASE/name).exists(),'Preserve existing attempt: '+name


def main():
    p=argparse.ArgumentParser();p.add_argument('--final-review',type=Path);p.add_argument('--root-selected',action='store_true');a=p.parse_args()
    assert a.root_selected,'Parent selection required; no implicit run'
    fresh();request=BASE/'request.json';receipt=BASE/'launch_receipt.json'
    if a.final_review:
        review=read(a.final_review);assert review['passed'] is True
        for role,path in [('request_subject',request),('launch_receipt_subject',receipt)]:
            assert Path(review[role]['path']).resolve()==path.resolve() and review[role]['sha256']==sha(path)
        write(BASE/'launch_clearance.json',dict(root_selected_single_saved_audit=True,request_sha256=sha(request),
            launch_receipt_sha256=sha(receipt),review={'path':a.final_review.resolve().as_posix(),'sha256':sha(a.final_review)},automatic_retry=False))
        return
    assert not receipt.exists() and not (BASE/'run_audit_durable.ps1').exists()
    q=read(request);assert q['kind']=='independent_saved_clock_audit' and q['source_review']['sha256']==SOURCE_REVIEW_SHA
    assert sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA
    review=read(SOURCE_REVIEW);assert review['passed'] is True and review['source_sha256']==q['source_sha256']
    for name,digest in q['source_sha256'].items():assert sha(SOURCE/name)==digest
    prep=read(BASE/'helper_preparation.json');assert prep['passed'] is True
    for name,digest in prep['helper_sha256'].items():assert sha(BASE/name)==digest
    pins={}
    def add(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None:assert digest==expected,path
        key=path.as_posix()
        if key in pins:assert pins[key]==digest
        pins[key]=digest
    for path,digest in q['input_sha256'].items():add(path,digest)
    for name,digest in prep['helper_sha256'].items():add(BASE/name,digest)
    for path in (request,BASE/'helper_preparation.json',BASE/'preserved_saved_audit_template.ps1.txt',
                 BASE/'template_parse.json',SOURCE_REVIEW,Path('C:/Windows/System32/wsl.exe')):add(path)
    target=BASE/'run_audit_durable.ps1'
    with target.open('x',encoding='utf-8') as f:f.write(render(sha(request)))
    add(target)
    write(receipt,dict(selected_single_saved_audit=True,request_sha256=sha(request),source_review_sha256=SOURCE_REVIEW_SHA,
        input_sha256=pins,wsl_path='C:/Windows/System32/wsl.exe',final_concrete_review_required=True,
        model_calls=0,native_steps=0,optimizer_updates=0,worker_processes=0,automatic_retry=False,dispatch_performed=False))
    print(json.dumps({'request_sha256':sha(request),'launch_receipt_sha256':sha(receipt),'pins':len(pins),'dispatched':False}))


if __name__=='__main__':main()
