"""Saved status only; no process/native calls. A diagnostic failure stays failure."""
import argparse,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
from packet_io import read,write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=BASE);args=parser.parse_args()
    output=args.base/'run';process=args.base/'clock_process';passed=False;error=None;report=None
    try:
        raw=read(process/'raw_exit.json');report=read(output/'report.json')
        stages=[read(p) for p in sorted((args.base/'stage_receipts').glob('*.json'))]
        preservation_complete=bool(stages and stages[-1]['stage']=='preservation_complete' and
            stages[-1]['report_sha256']==sha(output/'report.json') and
            stages[-1]['output_manifest_sha256']==sha(output/'output_manifest.json') and
            all(r['request_sha256']==sha(args.base/'clock_request.json') for r in stages))
        passed=bool(raw['known'] is True and raw['raw_python_exit_code']==0 and raw['raw_error'] is None and
                    preservation_complete and report['component_preliminary_pass'] is True and report['worker_pass'] is True and
                    report['all_four_MJB_exact'] is True and report['pre_and_post_input_hashes_exact'] is True)
    except Exception as exc:error=repr(exc)
    write(process/'diagnostic_verdict.json',{'diagnostic_passed':passed,'component_qualified':False,
        'root_saved_audit_pending':True,'error':error,'request_sha256':sha(args.base/'clock_request.json'),
        'report_sha256':sha(output/'report.json') if (output/'report.json').exists() else None})
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
