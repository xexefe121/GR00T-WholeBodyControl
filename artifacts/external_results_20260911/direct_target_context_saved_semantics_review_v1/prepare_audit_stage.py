"""Prepare a future actual saved audit; never dispatch and never create task data."""
import argparse
import re
from saved_common import BASE,NEW,RUN,sha,read,write,local,contains

OLD=NEW/'direct_target_full_state_saved_semantics_review_v1'
REVIEW=NEW/'direct_target_context_saved_semantics_root_review_v1/review.json'
REVIEW_SHA='73517c40584eeda0f7982f0f0653da3aecd949fc28dda85b091d9431799db89d'
WSL='C:/Windows/System32/wsl.exe'
BOOT='Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
INVENTORY_SHA='4296f2429a3483797c90589213f466e6912801bb1ee05f4f2c158caa69c7b07f'
VENV='E:/codex_sonic_runtime/mjbatch323_20260910/venv/'
HELPERS=('prepare_audit_stage.py','verify_completion.py','test_audit_stage.py','record_stage_preparation.py')


def one(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def render(request_sha):
    assert re.fullmatch('[0-9a-f]{64}',request_sha)
    text=(BASE/'preserved_run_audit_template.ps1.txt').read_text()
    text=one(text,'bfb87d2c4dde5e058dd15795c25407d1bd5c914f89af60e62f5499511b4d4898',request_sha)
    old='direct_target_full_state_saved_semantics_review_v1'
    assert text.count(old)==2
    text=text.replace(old,BASE.name)
    text=one(text,"if(Test-Path -LiteralPath (Join-Path $runRoot 'results_v1'))", """if($receipt.request_sha256 -ne (Get-TaskSha (Join-Path $runRoot 'request.json'))){throw 'Actual audit request changed.'}
$clearancePath=Join-Path $runRoot 'launch_clearance.json'
$clearance=Read-TaskJson $clearancePath
$clearanceSha=Get-TaskSha $clearancePath
if($clearance.root_selected_single_saved_audit -ne $true -or $clearance.request_sha256 -ne $receipt.request_sha256 -or $clearance.launch_receipt_sha256 -ne $LaunchReceiptSha256){throw 'Concrete saved-audit clearance required.'}
if((Get-TaskSha $clearance.review.path) -ne $clearance.review.sha256){throw 'Concrete review changed.'}
$review=Read-TaskJson $clearance.review.path
$slash=[string][char]47;$backslash=[string][char]92
if($review.passed -ne $true -or $review.request_subject.sha256 -ne $receipt.request_sha256 -or $review.request_subject.path -ne (Join-Path $runRoot 'request.json').Replace($backslash,$slash) -or $review.launch_receipt_subject.sha256 -ne $LaunchReceiptSha256 -or $review.launch_receipt_subject.path -ne $receiptPath.Replace($backslash,$slash)){throw 'Concrete review subject mismatch.'}
if(Test-Path -LiteralPath (Join-Path $runRoot 'results_v1'))""")
    text=one(text,'source_review_sha256=$receipt.source_review_sha256;task_model_calls=0',
        'source_review_sha256=$receipt.source_review_sha256;clearance_sha256=$clearanceSha;concrete_review_sha256=$clearance.review.sha256;task_model_calls=0')
    text=one(text,"if($result.evidence_audit_passed -ne $true){throw 'Saved evidence audit did not pass.'}",
        "if($result.evidence_audit_passed -ne $true -or $result.request_sha256 -ne $receipt.request_sha256 -or $result.context_condition -ne 'causal' -or $result.features -ne 1323 -or $result.model_calls -ne 0 -or $result.native_steps -ne 0){throw 'Saved evidence audit did not pass exact request/scope.'}")
    text=one(text,"$post=Check-Pins $receipt.input_sha256\n    Write-TaskJson", "$post=Check-Pins $receipt.input_sha256\n    if((Get-TaskSha $clearancePath) -ne $clearanceSha -or (Get-TaskSha $clearance.review.path) -ne $clearance.review.sha256){$runExit=1;$errorText='Concrete clearance/review changed.'}\n    Write-TaskJson")
    text=one(text,'launch_receipt_sha256=$LaunchReceiptSha256;automatic_retry=$false',
        'launch_receipt_sha256=$LaunchReceiptSha256;clearance_sha256=$clearanceSha;concrete_review_sha256=$clearance.review.sha256;automatic_retry=$false')
    return text


def runtime_entries(inventory):
    selected=[]
    for entry in inventory['files']:
        path=entry['path'];tail=path[len(VENV):] if path.startswith(VENV) else ''
        if tail=='pyvenv.cfg' or (tail.startswith('lib/python3.11/site-packages/') and
            tail.split('site-packages/',1)[1].split('/',1)[0] in ('numpy','numpy.libs','scipy','scipy.libs')):
            selected.append(entry)
    assert selected
    return selected


def require_fresh():
    for name in ('process_v1','results_v1','launch_clearance.json','owner_completion.json','dispatch.json'):
        assert not (BASE/name).exists(),'Preserve prior attempt: '+name


def finalize(path):
    require_fresh();request=BASE/'request.json';launch=BASE/'launch_receipt.json'
    review=read(path);assert review['passed'] is True
    for role,actual in [('request_subject',request),('launch_receipt_subject',launch)]:
        assert local(review[role]['path']).resolve()==actual.resolve() and review[role]['sha256']==sha(actual)
    write(BASE/'launch_clearance.json',dict(root_selected_single_saved_audit=True,request_sha256=sha(request),
        launch_receipt_sha256=sha(launch),review={'path':path.resolve().as_posix(),'sha256':sha(path)},automatic_retry=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-selected',action='store_true')
    parser.add_argument('--final-review',type=type(BASE));args=parser.parse_args()
    assert args.root_selected,'Actual saved audit must be selected by parent'
    if args.final_review:finalize(args.final_review);return
    require_fresh();assert not (BASE/'launch_receipt.json').exists() and not (BASE/'run_audit_durable.ps1').exists()
    request=read(BASE/'request.json');request_sha=sha(BASE/'request.json')
    assert request['context_condition']=='causal' and request['public_features']==1323
    assert sha(REVIEW)==REVIEW_SHA==request['source_review_sha256']
    review=read(REVIEW);prep=read(BASE/'source_preparation.json')
    assert review['source_review_pass'] is True and review['source_sha256']==prep['source_sha256']
    for name,digest in prep['source_sha256'].items():assert sha(BASE/name)==digest,name
    stage=read(BASE/'stage_source_preparation.json')
    for name,digest in stage['helper_sha256'].items():assert sha(BASE/name)==digest,name
    assert stage['passed'] is True and stage['synthetic_tests_passed'] is True
    pins={}
    def add(path,digest=None):
        path=local(path).resolve();actual=sha(path)
        if digest is not None:assert actual==digest,path
        name=path.as_posix()
        if name in pins:assert pins[name]==actual
        pins[name]=actual
    for path,digest in request['input_sha256'].items():add(path,digest)
    for name,digest in stage['helper_sha256'].items():add(BASE/name,digest)
    for path in (BASE/'request.json',BASE/'source_preparation.json',BASE/'stage_source_preparation.json',
                 BASE/'preserved_run_audit_template.ps1.txt',BASE/'stage_template_parse.json',REVIEW,WSL,BOOT):add(path)
    inventory_path=RUN/'runtime_inventory.json';add(inventory_path,INVENTORY_SHA)
    runtime=runtime_entries(read(inventory_path))
    for entry in runtime:add(entry['path'],entry['sha256'])
    actual_script=BASE/'run_audit_durable.ps1'
    with actual_script.open('x',encoding='utf-8') as stream:stream.write(render(request_sha))
    add(actual_script)
    receipt=dict(selected_single_saved_audit=True,request_sha256=request_sha,source_review_sha256=REVIEW_SHA,
        input_sha256=pins,wsl_path=WSL,context_condition='causal',features=1323,
        runtime_package_files=len(runtime),runtime_inventory_sha256=INVENTORY_SHA,
        final_concrete_review_required=True,model_calls=0,native_steps=0,optimizer_updates=0,automatic_retry=False,
        actual_dispatch_performed=False,canonical_evaluation_cleared=False)
    write(BASE/'launch_receipt.json',receipt)
    print({'request_sha256':request_sha,'launch_receipt_sha256':sha(BASE/'launch_receipt.json'),'pins':len(pins),'dispatched':False})


if __name__=='__main__':main()
