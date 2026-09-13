"""Future saved owner accounting. No native/model calls or automatic retry."""
import json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
from packet_io import read,write,sha,pin_check
from sidecar_owner_evidence import check_sidecars


def absent_processes(folder,output):
    start=read(folder/'start.json');child=read(folder/'child.json')
    windows=[int(start['wrapper_pid']),int(child['child_pid'])]
    command='@('+','.join(str(v) for v in windows)+") | ForEach-Object { if (Get-Process -Id $_ -ErrorAction SilentlyContinue) { Write-Output $_ } }"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    present_windows=[int(v) for v in result.stdout.split()]
    linux=[]
    if (output/'process_start.json').exists():linux.append(int(read(output/'process_start.json')['pid']))
    if (output/'worker_ready.json').exists():
        record=read(output/'worker_ready.json');linux.append(int(record['worker_pid']))
        if record.get('resource_tracker_pid') is not None:linux.append(int(record['resource_tracker_pid']))
    linux=sorted(set(linux));script=' '.join('test ! -d /proc/'+str(v)+' || echo '+str(v)+';' for v in linux)
    present_linux=[]
    if linux:
        result=subprocess.run(['wsl','-d','Ubuntu-22.04','--cd','/','--','sh','-c',script],capture_output=True,text=True,check=True)
        present_linux=[int(v) for v in result.stdout.split()]
    return {'windows_expected_pids':windows,'windows_present':present_windows,'linux_expected_pids':linux,
            'linux_present':present_linux,'all_observed_pids_absent':not present_windows and not present_linux,
            'no_claim_for_unobserved_failed_start_descendants':True}


def main():
    folder=BASE/'clock_process';output=BASE/'run';receipt_path=folder/'launch_receipt.json';receipt=read(receipt_path)
    clearance=read(folder/'launch_clearance.json');exit_record=read(folder/'exit.json');raw=read(folder/'raw_exit.json')
    diagnostic=read(folder/'diagnostic_verdict.json');start=read(folder/'start.json');child=read(folder/'child.json')
    assert clearance['launch_receipt_sha256']==sha(receipt_path) and clearance['request_sha256']==receipt['request_sha256']
    assert sha(clearance['review']['path'])==clearance['review']['sha256']
    assert start['receipt_sha256']==sha(receipt_path) and start['clearance_sha256']==sha(folder/'launch_clearance.json')
    assert start['review_sha256']==clearance['review']['sha256'] and child['wrapper_pid']==start['wrapper_pid'] and child['handle_acquired'] is True
    assert exit_record['known'] is True and raw['known'] is True and raw['raw_error'] is None
    assert exit_record['raw_python_exit_code']==raw['raw_python_exit_code']
    # Known nonzero physical/timing verdict can have complete, honest accounting.
    assert exit_record['diagnostic_exit_code']==(0 if diagnostic['diagnostic_passed'] else 2)
    assert exit_record['exit_code']==exit_record['diagnostic_exit_code'] and exit_record['error'] is None
    for name in ('prerun_hashes.json','postrun_hashes.json'):
        checks=read(folder/name);assert checks['all_exact'] is True and set(checks['files'])==set(receipt['input_hashes'])
        for path,digest in receipt['input_hashes'].items():assert checks['files'][path]=={'expected':digest,'actual':digest,'matched':True}
    current=pin_check([{'path':k,'sha256':v} for k,v in receipt['input_hashes'].items()]);assert current['all_exact']
    report=read(output/'report.json');manifest=read(output/'output_manifest.json')
    sidecars=check_sidecars(output,report,manifest)
    assert diagnostic['timing_sidecar_accounting']==sidecars
    assert report['request_sha256']==manifest['request_sha256']==receipt['request_sha256']
    assert report['component_preliminary_pass'] is diagnostic['diagnostic_passed']
    outputs={}
    for relative,entry in manifest['files'].items():
        path=(output/relative).resolve();assert path.is_relative_to(output.resolve()) and sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes']
        outputs[path.as_posix()]=sha(path)
    stages=sorted((BASE/'stage_receipts').glob('*.json'));assert stages
    for path in stages:
        record=read(path);assert record['request_sha256']==receipt['request_sha256'];outputs[path.as_posix()]=sha(path)
    last=read(stages[-1]);assert last['stage']=='preservation_complete'
    assert last['report_sha256']==sha(output/'report.json') and last['output_manifest_sha256']==sha(output/'output_manifest.json')
    absence=absent_processes(folder,output);write(folder/'process_absence.json',absence)
    cleanup=report['worker_cleanup'] or {}
    absence_proven=absence['all_observed_pids_absent'] and not cleanup.get('start_side_effects_uncertain',False)
    for path in folder.glob('*'):
        if path.is_file():outputs[path.as_posix()]=sha(path)
    outputs[(output/'output_manifest.json').as_posix()]=sha(output/'output_manifest.json')
    result={'passed':bool(absence_proven),'accounting_passed':True,'process_absence_proven':absence_proven,
        'diagnostic_passed':diagnostic['diagnostic_passed'],'timing_sidecar_accounting':sidecars,'component_qualified':False,'root_saved_audit_pending':True,
        'request_sha256':receipt['request_sha256'],'report_sha256':sha(output/'report.json'),
        'process_exit_sha256':sha(folder/'exit.json'),'launch_receipt_sha256':sha(receipt_path),
        'clearance_sha256':sha(folder/'launch_clearance.json'),'process_absence_sha256':sha(folder/'process_absence.json'),
        'pre_and_post_input_hashes_exact':True,'raw_python_exit_code':raw['raw_python_exit_code'],
        'input_hashes':receipt['input_hashes'],'output_hashes':outputs,'native_steps':0,'model_calls':0}
    write(BASE/'owner_completion.json',result);print(json.dumps({'owner_sha256':sha(BASE/'owner_completion.json'),
        'accounting_passed':True,'process_absence_proven':absence_proven,'diagnostic_passed':diagnostic['diagnostic_passed']}))


if __name__=='__main__':main()
