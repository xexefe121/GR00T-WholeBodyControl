"""Known-timeout preservation only. No completion claim, native call or retry."""
import datetime,hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
FOLDER=BASE/'clock_process';OUT=BASE/'timeout_preservation_v1'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def utc(value):return datetime.datetime.fromtimestamp(value,datetime.timezone.utc).isoformat()
def main():
    OUT.mkdir(exist_ok=False)
    request=read(BASE/'clock_request.json');receipt=read(FOLDER/'launch_receipt.json');start=read(FOLDER/'start.json');child=read(FOLDER/'child.json')
    exit_record=read(FOLDER/'exit.json');raw=read(FOLDER/'raw_exit.json');diagnostic=read(FOLDER/'diagnostic_verdict.json')
    assert raw['known'] is True and raw['raw_python_exit_code']==124 and raw['raw_error'] is None
    assert exit_record['known'] is True and exit_record['raw_python_exit_code']==124 and exit_record['diagnostic_exit_code']==exit_record['exit_code']==2
    assert exit_record['error'] is None and exit_record['all_postrun_hashes_exact'] is True and diagnostic['diagnostic_passed'] is False
    assert not (BASE/'run/report.json').exists() and not (BASE/'run/trace.npz').exists()
    for name in ('prerun_hashes.json','postrun_hashes.json'):
        record=read(FOLDER/name);assert record['all_exact'] is True and set(record['files'])==set(receipt['input_hashes'])
        for path,digest in receipt['input_hashes'].items():assert record['files'][path]==dict(expected=digest,actual=digest,matched=True)
    pins={path:dict(expected=digest,actual=sha(Path(path))) for path,digest in receipt['input_hashes'].items()}
    assert all(v['expected']==v['actual'] for v in pins.values());write(OUT/'current_input_hashes.json',pins)
    linux_start=read(BASE/'run/process_start.json');worker=read(BASE/'run/worker_ready.json')
    windows=[start['wrapper_pid'],child['child_pid'],8448];linux=[linux_start['pid'],worker['resource_tracker_pid'],worker['worker_pid']]
    assert windows==[26068,16024,8448] and linux==[390,465,466]
    ps=subprocess.run(['powershell','-NoProfile','-Command','Get-Process -Id 26068,16024,8448 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id'],capture_output=True,text=True)
    native_ps=subprocess.run(['wsl','-d','Ubuntu-22.04','--cd','/','--','/bin/ps','-p','390,465,466','-o','pid=,ppid=,stat=,args='],capture_output=True,text=True)
    present_windows=[int(v) for v in ps.stdout.split()];present_linux=[line for line in native_ps.stdout.splitlines() if line.strip()]
    assert not present_windows and not present_linux and native_ps.returncode in (0,1)
    absence=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),windows=dict(expected_pids=windows,present=present_windows,returncode=ps.returncode,stdout=ps.stdout,stderr=ps.stderr),
        linux=dict(expected_pids=linux,present=present_linux,returncode=native_ps.returncode,stdout=native_ps.stdout,stderr=native_ps.stderr),
        namespaces='Windows wrapper26068/child16024/wsl8448; Linux native390/resource_tracker465/worker466',all_observed_processes_absent=True)
    write(OUT/'process_absence.json',absence)
    files={}
    for folder in (BASE/'run',FOLDER):
        for path in folder.rglob('*'):
            if path.is_file():
                stat=path.stat();files[path.relative_to(BASE).as_posix()]=dict(sha256=sha(path),bytes=stat.st_size,creation_utc=utc(stat.st_ctime),modified_utc=utc(stat.st_mtime))
    mjb=[]
    for path in sorted((BASE/'run/mjb').glob('*.mjb')):
        digest=sha(path);assert digest==request['expected_model_sha256'];mjb.append(dict(path=path.as_posix(),sha256=digest,bytes=path.stat().st_size))
    assert len(mjb)==2
    start_utc=datetime.datetime.fromisoformat(start['utc'].replace('Z','+00:00'));raw_utc=datetime.datetime.fromisoformat(raw['utc'].replace('Z','+00:00'))
    output_start=datetime.datetime.fromtimestamp((BASE/'run/process_start.json').stat().st_mtime,datetime.timezone.utc)
    worker_ready=datetime.datetime.fromtimestamp((BASE/'run/worker_ready.json').stat().st_mtime,datetime.timezone.utc)
    result=dict(kind='known_timeout_partial_preservation',preservation_accounting_passed=True,completed_owner_accounting=False,component_qualified=False,
        raw_python_exit_code=124,diagnostic_exit_code=2,original_requested_controls=1819,original_requested_native_steps=18190,
        actual_native_attempted=None,actual_native_returned=None,actual_captured_samples=None,actual_counts_unavailable=True,
        committed_trace_file_present=False,final_report_present=False,epoch_timestamp_retained=False,exit_model_identity_present=False,
        preserved_entry_mjb_buffers=mjb,all_observed_processes_absent=True,process_absence_sha256=sha(OUT/'process_absence.json'),
        timing_from_file_metadata=dict(wrapper_start_to_run_setup_receipt_seconds=(output_start-start_utc).total_seconds(),
            run_setup_receipt_to_worker_ready_seconds=(worker_ready-output_start).total_seconds(),worker_ready_to_timeout_return_seconds=(raw_utc-worker_ready).total_seconds(),
            wrapper_start_to_timeout_return_seconds=(raw_utc-start_utc).total_seconds(),fixed_full_clock_duration_seconds=36.38,
            limitation='Filesystem timestamps locate broad stages; no retained epoch/step ledger proves exact execution counts.'),
        startup_receipts=dict(parent=linux_start,worker=worker),available_files=files,
        request_sha256=sha(BASE/'clock_request.json'),launch_receipt_sha256=sha(FOLDER/'launch_receipt.json'),clearance_sha256=sha(FOLDER/'launch_clearance.json'),
        exit_sha256=sha(FOLDER/'exit.json'),source_sha256=sha(Path(__file__)),current_input_hashes_sha256=sha(OUT/'current_input_hashes.json'),
        native_calls_in_this_check=0,model_calls=0,repeated_benchmark=False,
        limitation='Timeout killed in-memory evidence before final report/trace. No final saved-array audit can run and no full clock/physics claim is made.')
    write(OUT/'report.json',result)
    print(json.dumps(dict(report_sha256=sha(OUT/'report.json'),processes_absent=True,input_pins=len(pins),preserved_mjb=2,timing=result['timing_from_file_metadata'])))
if __name__=='__main__':main()
