"""Freeze selected prefix capture and prepare hidden durable scripts; no launch."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).parent;SOURCE=BASE/'source_snapshot_v1';NEW=BASE.parent;OLD=NEW.parent/'sonic23_teleop_six_hour_20260910'
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_new(path,text):
    with Path(path).open('x',encoding='utf-8') as f:f.write(text)
def quote(value):return "'"+str(value).replace("'","''")+"'"

prior=read(NEW/'fast_controller_phase_fit_v1/frozen_inputs_v2.json')
pins=dict(prior['input_sha256'])
for path,digest in pins.items():assert sha(path)==digest,path
for name,digest in prior['source_sha256'].items():assert sha(SOURCE/name)==digest,name
for path in SOURCE.rglob('*.py'):pins[path.as_posix()]=sha(path)
old=OLD/'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1'
bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
fixture=NEW/'walk003_canonical_initial_fixture_v1/initial_integration_state.npz'
extra=[old/'trace.npz',old/'report.json',old/'provenance.json',old/'recorded_source_audit_v2.json',
    OLD/'bfm_online_intent_v2/walk003_quiet_frozen_import_verification_v3/report.json',
    fixture,fixture.parent/'report.json',BASE/'upstream_reconstruct_control_snapshots.py',BASE/'source_derivation.json',
    BASE/'prepare_source.py',BASE/'test_capture.py',BASE/'focused.xml',Path(__file__),
    NEW/'broader_labels_independent_v1/exit_capture_test.json',
    ROOT/'artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh']
for path in extra:pins[path.as_posix()]=sha(path)
assert read(extra[-2])['actual_exit']==read(extra[-2])['expected_exit']==7
request=dict(kind='selected_old_expert_recorded_prefix_full291_snapshot_preparation',root_selected=True,
    prefix_controls=1268,requested_native_steps=12680,requested_snapshots=1269,boundary_controls=[0,1268],
    last_snapshot_is_unexecuted_control=True,original_full_lifecycle_controls=1569,clip='walk003',
    trace=(old/'trace.npz').as_posix(),trace_sha256=sha(old/'trace.npz'),original_report=(old/'report.json').as_posix(),
    fixture=fixture.as_posix(),fixture_report=(fixture.parent/'report.json').as_posix(),bundle=bundle.as_posix(),
    source_warning_serialization='int64',native_warning_type='int32',
    full291_endpoint_previously_available=False,source_state_rewrites_after_initialization=0,
    inference_authorized=False,labels_authorized=False,perturbations_authorized=False,hardware_authorized=False,input_hashes=pins)
request_path=BASE/'capture_request.json';write_new(request_path,json.dumps(request,indent=2)+'\n')
process=BASE/'capture_process';process.mkdir(exist_ok=False)
linux='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/'+BASE.name
arguments=['-d','Ubuntu-22.04','--cd','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof','--','bash',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH='+linux+'/source_snapshot_v1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',linux+'/source_snapshot_v1/capture_old_prefix.py']
HASH="""$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
"""
launcher=process/'run.ps1';durable=process/'run_durable.ps1'
write_new(launcher,HASH+'$requestPath = '+quote(request_path)+'\n'+
    'if ((Get-TaskHash $requestPath) -ne '+quote(sha(request_path))+') { throw \'Capture request changed.\' }\n'+
    '$request = Get-Content -Raw -LiteralPath $requestPath | ConvertFrom-Json\n'+
    "foreach ($entry in $request.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Capture input changed: ' + $entry.Name) } }\n"+
    '$arguments = @(\n'+',\n'.join('    '+quote(a) for a in arguments)+'\n)\n'+
    '& "$env:SystemRoot\\System32\\wsl.exe" @arguments\n$taskExit = $LASTEXITCODE\n'+
    "if ($null -eq $taskExit) { throw 'WSL exit status unavailable.' }\nif ($taskExit -ne 0) { exit $taskExit }\n"+
    '$result = Get-Content -Raw -LiteralPath '+quote(BASE/'capture/report.json')+' | ConvertFrom-Json\n'+
    "if (-not $result.passed -or $result.completed_controls -ne 1268 -or $result.completed_native_steps -ne 12680 -or $result.verified_native_steps -ne 12680 -or $result.snapshots -ne 1269 -or $result.new_inference_calls -ne 0 -or $result.new_labels -ne 0 -or $result.new_perturbations -ne 0 -or $result.original_full_lifecycle_reexecuted -or $result.last_snapshot_control_executed) { throw 'Incomplete or out-of-scope prefix capture.' }\nexit 0\n")
write_new(durable,HASH+'$folder = '+quote(process)+'\n'+
    "$receiptPath = Join-Path $folder 'launch_receipt.json'\n$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json\n"+
    "foreach ($entry in $receipt.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Launch input changed: ' + $entry.Name) } }\n"+
    "$lock = [System.IO.File]::Open((Join-Path $folder 'started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)\n$lock.Close()\n"+
    "[ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;receipt_sha256=(Get-TaskHash $receiptPath);requested_controls=1268;requested_snapshots=1269} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'start.json') -Encoding UTF8\n"+
    "$exitCode = 1\n$errorText = $null\ntry {\n"+
    "    $taskChild = Start-Process -FilePath \"$env:SystemRoot\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'stdout.log') -RedirectStandardError (Join-Path $folder 'stderr.log')\n"+
    "    $nativeHandle = $taskChild.Handle\n"+
    "    [ordered]@{child_pid=$taskChild.Id;wrapper_pid=$PID;handle_acquired=($nativeHandle -ne [IntPtr]::Zero)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'child.json') -Encoding UTF8\n"+
    "    $taskChild.WaitForExit()\n    $exitCode = $taskChild.ExitCode\n    if ($null -eq $exitCode) { throw 'Child exit status unavailable.' }\n"+
    "    $post = [ordered]@{}\n    foreach ($entry in $receipt.input_hashes.PSObject.Properties) { $actual = Get-TaskHash $entry.Name; $post[$entry.Name] = $actual; if ($actual -ne $entry.Value) { throw ('Postrun input changed: ' + $entry.Name) } }\n"+
    "    $post | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $folder 'postrun_hashes.json') -Encoding UTF8\n"+
    "} catch { $errorText=$_.Exception.Message; $exitCode=1 } finally {\n"+
    "    [ordered]@{utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitCode;error=$errorText} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'exit.json') -Encoding UTF8\n}\nexit $exitCode\n")
launch_pins=dict(pins)
for path in (request_path,launcher,durable):launch_pins[path.as_posix()]=sha(path)
receipt=dict(kind='one_selected_old_expert1268_control_snapshot_prefix',requested_controls=1268,requested_native_steps=12680,
    requested_boundary_snapshots=1269,inference=False,labels=False,perturbations=False,hardware=False,
    capture_request_sha256=sha(request_path),exact_wsl_arguments=arguments,input_hashes=launch_pins)
write_new(process/'launch_receipt.json',json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(request_sha256=sha(request_path),request_pins=len(pins),launch_receipt_sha256=sha(process/'launch_receipt.json'),launch_pins=len(launch_pins),launched=False)))
