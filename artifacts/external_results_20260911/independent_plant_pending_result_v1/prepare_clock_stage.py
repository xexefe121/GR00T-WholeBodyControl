"""Build hidden fixed command and receipt. Selection remains a separate action."""
import argparse,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
from packet_io import sha,read,write,pin_check
import prepare_stage_preserved_template as template
WINPY=template.WINPY


def linux(path):return template.linux(path)


def arguments(stage):
    if stage!='clock':raise ValueError('single bounded clock stage only')
    return ['-d','Ubuntu-22.04','--cd','/','--','bash',linux(template.BOOT),
        'timeout','--signal=TERM','--kill-after=5s','555s','env',
        'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1',
        'PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+linux(BASE/'source_draft_v1'),template.PY,'-B',
        linux(BASE/'source_draft_v1/run_clock.py'),'--request',linux(BASE/'clock_request.json'),
        '--clearance',linux(BASE/'clock_process/launch_clearance.json')]


def texts():
    # Preserve established exact handle/known-exit/CreateNew/no-retry wrapper;
    # only streamed hashing, stage arguments and saved-verdict CLI differ.
    template.arguments=arguments
    template.PS_HELPERS=template.PS_HELPERS.replace(
        "try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }\n    finally { $algorithm.Dispose() }",
        """$stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }""")
    run=template.run_text('clock');durable=template.durable_text('clock')
    durable=durable.replace(" --stage 'clock'",'')
    if 'ReadAllBytes' in run+durable:raise ValueError('streaming hash replacement did not apply')
    return run,durable


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final-review',type=Path)
    parser.add_argument('--pass-field',default='passed');parser.add_argument('--root-selected',action='store_true');args=parser.parse_args()
    request_path=BASE/'clock_request.json';folder=BASE/'clock_process';receipt_path=folder/'launch_receipt.json'
    if args.final_review:
        if not args.root_selected:raise ValueError('parent must select this actual one run')
        review=read(args.final_review)
        if review[args.pass_field] is not True:raise ValueError('final review failed')
        for key,path in [('request_subject',request_path),('launch_receipt_subject',receipt_path)]:
            if Path(review[key]['path']).resolve()!=path.resolve() or review[key]['sha256']!=sha(path):
                raise ValueError('final review literal subject differs')
        if (folder/'started.lock').exists() or (BASE/'run').exists() or (BASE/'stage_receipts').exists():raise ValueError('preserve prior attempt')
        write(folder/'launch_clearance.json',{'root_selected_single_run':True,'request_sha256':sha(request_path),
            'launch_receipt_sha256':sha(receipt_path),'review':{'path':args.final_review.resolve().as_posix(),
            'sha256':sha(args.final_review),'pass_field':args.pass_field},'automatic_retry':False,'hardware_authorized':False})
        print(json.dumps({'clearance_sha256':sha(folder/'launch_clearance.json'),'launched':False}));return
    if args.root_selected:raise ValueError('source preparation does not select actual execution')
    if folder.exists() or (BASE/'run').exists() or (BASE/'stage_receipts').exists():raise ValueError('fresh stage output required')
    request=read(request_path)
    from stage_watchdog import validate_request
    validate_request(request)
    from prepare_concrete_packet import validate_retry_contract
    validate_retry_contract(request)
    if request['outer_process_timeout_seconds']!=555:raise ValueError('literal outer watchdog coverage required')
    checks=pin_check(request['input_files'])
    if not checks['all_exact']:raise ValueError('request input changed')
    pins={entry['path']:entry['sha256'] for entry in request['input_files']}
    for path in (request_path,Path(__file__),BASE/'prepare_stage_preserved_template.py',BASE/'stage_verdict.py',BASE/'verify_completion.py',Path(WINPY)):
        pins[path.resolve().as_posix()]=sha(path)
    folder.mkdir();run,durable=texts()
    for path,text in [(folder/'run.ps1',run),(folder/'run_durable.ps1',durable)]:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)
        pins[path.resolve().as_posix()]=sha(path)
    write(receipt_path,{'stage':'clock','request_path':request_path.as_posix(),'request_sha256':sha(request_path),
        'exact_wsl_arguments':arguments('clock'),'input_hashes':pins,'requested_native_steps':18190,
        'requested_serializations':4,'execution_selected':False,'final_clearance_required':True,
        'automatic_retry':False,'hardware_authorized':False})
    print(json.dumps({'request_sha256':sha(request_path),'launch_receipt_sha256':sha(receipt_path),'pins':len(pins),'launched':False}))


if __name__=='__main__':main()
