"""Saved source/hash/test receipts only. No task dynamics or process benchmark."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source_runner_v1'))
from packet_io import read,write,sha,pin_check


def main():
    request=read(BASE/'clock_request.json');receipt=read(BASE/'clock_process/launch_receipt.json')
    checks=pin_check([{'path':k,'sha256':v} for k,v in receipt['input_hashes'].items()]);assert checks['all_exact']
    assert receipt['request_sha256']==sha(BASE/'clock_request.json')
    suites=ET.parse(BASE/'runner_stub_tests.xml').getroot();cases=suites.findall('.//testcase')
    assert len(cases)==30 and not suites.findall('.//failure') and not suites.findall('.//error') and not suites.findall('.//skipped')
    parse=read(BASE/'runner_powershell_parse.json');assert len(parse)==2 and all(not row['errors'] for row in parse)
    absent=['run','clock_process/started.lock','clock_process/launch_clearance.json','clock_process/start.json','clock_process/exit.json']
    assert all(not (BASE/name).exists() for name in absent)
    source_hashes={p.name:sha(p) for p in (BASE/'source_runner_v1').glob('*.py')}
    copies=request['byte_preserved_copies']
    assert all(source_hashes[name]==entry['sha256']==sha(entry['original_path']) for name,entry in copies.items())
    evidence={p.as_posix():sha(p) for p in [BASE/'clock_request.json',BASE/'clock_process/launch_receipt.json',
        BASE/'runner_stub_tests.xml',BASE/'runner_powershell_parse.json',BASE/'OUTPUT_SCHEMA.md',Path(__file__)]}
    result={'kind':'process_clock_concrete_source_preparation','preparation_passed':True,'execution_selected':False,
        'actual_native_steps':0,'actual_MJB_serializations':0,'actual_worker_starts':0,'actual_model_calls':0,
        'planned_native_steps':18190,'planned_serializations':4,'planned_main_controls':1569,'planned_hold_controls':250,
        'stub_tests':30,'skipped':0,'source_hashes':source_hashes,'byte_preserved_copies':copies,
        'request_sha256':sha(BASE/'clock_request.json'),'launch_receipt_sha256':sha(BASE/'clock_process/launch_receipt.json'),
        'launch_pin_count':len(receipt['input_hashes']),'all_launch_pins_exact':True,'evidence_hashes':evidence,
        'absent_execution_artifacts':absent,'root_saved_audit_required':True,'limitations_document':'OUTPUT_SCHEMA.md'}
    write(BASE/'runner_source_preparation.json',result)
    print(json.dumps({'preparation_sha256':sha(BASE/'runner_source_preparation.json'),
        'request_sha256':result['request_sha256'],'launch_receipt_sha256':result['launch_receipt_sha256'],
        'source_files':len(source_hashes),'copied_unchanged':len(copies),'stub_tests':30,'launch_pins':result['launch_pin_count']}))


if __name__=='__main__':main()
