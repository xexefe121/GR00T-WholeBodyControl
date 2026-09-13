"""Owner saved-file/process-receipt accounting, never a native or process launch."""
import argparse,hashlib,json
from pathlib import Path
from stage_verdict import read,validate_report

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['witness','replay'],required=True)
    p.add_argument('--process-check',type=Path,required=True);a=p.parse_args()
    base=Path(__file__).resolve().parent;folder=base/(a.stage+'_process')
    process=read(a.process_check);start=read(folder/'start.json');child=read(folder/'child.json')
    assert process['wrapper_absent'] is True and process['child_absent'] is True
    assert process['wrapper_pid']==start['wrapper_pid'] and process['child_pid']==child['child_pid']
    receipt_path=folder/'launch_receipt.json';receipt=read(receipt_path)
    pre=read(folder/'prerun_hashes.json');post=read(folder/'postrun_hashes.json')
    assert pre['all_exact'] is True and post['all_exact'] is True
    for record in (pre,post):
        assert set(record['files'])==set(receipt['input_hashes'])
        for path,digest in receipt['input_hashes'].items():
            assert record['files'][path]==dict(expected=digest,actual=digest,matched=True)
            assert sha(path)==digest,path
    clearance_path=folder/'launch_clearance.json';clearance=read(clearance_path)
    request_path=base/(a.stage+'_request.json');report_path=base/a.stage/'report.json'
    assert sha(receipt_path)==start['receipt_sha256']==clearance['launch_receipt_sha256']
    assert sha(request_path)==receipt['request_sha256']==clearance['request_sha256']
    assert sha(clearance_path)==start['clearance_sha256']
    assert sha(clearance['review']['path'])==clearance['review']['sha256']==start['review_sha256']
    raw=read(folder/'raw_exit.json');exit_record=read(folder/'exit.json');verdict=read(folder/'diagnostic_verdict.json')
    assert raw['known'] is True and raw['raw_error'] is None and raw['raw_python_exit_code']==0
    assert exit_record['known'] is True and exit_record['error'] is None and exit_record['all_postrun_hashes_exact'] is True
    assert exit_record['exit_code']==exit_record['raw_python_exit_code']==exit_record['diagnostic_exit_code']==0
    assert verdict['passed'] is True and verdict['raw_python_exit_code']==verdict['diagnostic_exit_code']==0
    report=read(report_path);validate_report(a.stage,report);assert report['request_sha256']==sha(request_path)
    outputs=[p for p in (base/a.stage).rglob('*') if p.is_file()]
    outputs.extend(p for p in folder.iterdir() if p.is_file());outputs.extend([a.process_check,Path(__file__),base/'stage_verdict.py'])
    result=dict(passed=True,owner_completion_accounting_passed=True,stage=a.stage,
        request_sha256=sha(request_path),report_sha256=sha(report_path),process_exit_sha256=sha(folder/'exit.json'),
        launch_receipt_sha256=sha(receipt_path),clearance_sha256=sha(clearance_path),
        pre_and_post_input_hashes_exact=True,process_absence=process,process_absence_sha256=sha(a.process_check),
        api_counters=report['api_counters'],input_hashes=receipt['input_hashes'],
        case_counters={k:report[k]['adapter_counters'] for k in ('expert','direct')} if a.stage=='replay' else {},
        output_hashes={x.resolve().as_posix():sha(x) for x in outputs},
        independent_saved_audit_pending=True,policy_balance_or_realtime_qualification=False)
    path=base/(a.stage+'_completion_verification.json')
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(path),sha256=sha(path),passed=True)))

if __name__=='__main__':main()
