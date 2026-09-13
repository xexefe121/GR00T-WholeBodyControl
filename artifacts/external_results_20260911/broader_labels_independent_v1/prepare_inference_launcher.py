"""Freeze one independent BFM reevaluation with verified process-handle capture."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).parent;PROCESS=BASE/'inference_process'
PROCESS.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quote(s):return "'"+str(s).replace("'","''")+"'"
request=BASE/'inference_request_v2.json';r=json.loads(request.read_text())
assert r['independent_inference_selected'] and not (BASE/'baseline_inference_audit').exists()
exit_test=json.loads((BASE/'exit_capture_test.json').read_text(encoding='utf-8-sig'))
assert exit_test['actual_exit']==exit_test['expected_exit']==7 and exit_test['handle_acquired_before_wait']
arguments=['-d','Ubuntu-22.04','--cd','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof','--','bash',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
    'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/broader_labels_independent_v1/source_inference_v2',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/broader_labels_independent_v1/source_inference_v2/audit.py',
    '--request','/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/broader_labels_independent_v1/inference_request_v2.json',
    '--phase','inference']
hash_function="""$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
"""
launcher=PROCESS/'run_audit.ps1'
launcher.write_text(hash_function+'$requestPath = '+quote(request)+'\n'+
    'if ((Get-TaskHash $requestPath) -ne '+quote(sha(request))+') { throw \'Audit request changed.\' }\n'+
    '$request = Get-Content -Raw -LiteralPath $requestPath | ConvertFrom-Json\n'+
    "foreach ($entry in $request.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Audit input changed: ' + $entry.Name) } }\n"+
    '$arguments = @(\n'+',\n'.join('    '+quote(a) for a in arguments)+'\n)\n'+
    '& "$env:SystemRoot\\System32\\wsl.exe" @arguments\n$taskExit = $LASTEXITCODE\n'+
    "if ($null -eq $taskExit) { throw 'WSL exit status unavailable.' }\nif ($taskExit -ne 0) { exit $taskExit }\n"+
    '$result = Get-Content -Raw -LiteralPath '+quote(BASE/'baseline_inference_audit/report.json')+' | ConvertFrom-Json\n'+
    "if (-not $result.pass_all -or $result.selected_rows_checked -ne 6847 -or $result.actor_inference_calls -ne 6847 -or $result.backward_inference_calls -ne 6847 -or $result.physics_steps -ne 0 -or -not $result.actual_BFM_output_authenticity_verified -or $result.fitting_launched) { throw 'Incomplete independent audit.' }\nexit 0\n")
durable=PROCESS/'run_audit_durable.ps1'
durable.write_text(hash_function+'$folder = '+quote(PROCESS)+'\n'+
    "$receiptPath = Join-Path $folder 'launch_receipt.json'\n$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json\n"+
    "foreach ($entry in $receipt.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Launch input changed: ' + $entry.Name) } }\n"+
    "$lock = [System.IO.File]::Open((Join-Path $folder 'started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)\n$lock.Close()\n"+
    "[ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;receipt_sha256=(Get-TaskHash $receiptPath);expected_actor=6847;expected_backward=6847} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'start.json') -Encoding UTF8\n"+
    "$exitCode = 1\n$errorText = $null\ntry {\n"+
    "    $taskChild = Start-Process -FilePath \"$env:SystemRoot\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run_audit.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'stdout.log') -RedirectStandardError (Join-Path $folder 'stderr.log')\n"+
    "    $nativeHandle = $taskChild.Handle\n"+
    "    [ordered]@{child_pid=$taskChild.Id;wrapper_pid=$PID;handle_acquired=($nativeHandle -ne [IntPtr]::Zero)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'child.json') -Encoding UTF8\n"+
    "    $taskChild.WaitForExit()\n    $exitCode = $taskChild.ExitCode\n    if ($null -eq $exitCode) { throw 'Child exit status unavailable.' }\n"+
    "    $post = [ordered]@{}\n    foreach ($entry in $receipt.input_hashes.PSObject.Properties) { $actual = Get-TaskHash $entry.Name; $post[$entry.Name] = $actual; if ($actual -ne $entry.Value) { throw ('Postrun input changed: ' + $entry.Name) } }\n"+
    "    $post | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $folder 'postrun_hashes.json') -Encoding UTF8\n"+
    "} catch { $errorText=$_.Exception.Message; $exitCode=1 } finally {\n"+
    "    [ordered]@{utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitCode;error=$errorText;expected_actor=6847;expected_backward=6847} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'exit.json') -Encoding UTF8\n}\nexit $exitCode\n")
pins=dict(r['input_hashes'])
for p in (request,Path(__file__),launcher,durable,BASE/'exit7_stub.ps1',BASE/'check_exit_capture.ps1',BASE/'exit_capture_test.json',
          Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh')):
    pins[str(p).replace('\\','/')]=sha(p)
receipt=dict(kind='one_selected_independent_6847_BFM_verification',expected_actor=6847,expected_backward=6847,
    new_labels=False,physics=False,fitting=False,exact_wsl_arguments=arguments,input_hashes=pins)
path=PROCESS/'launch_receipt.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(launch_receipt_sha256=sha(path),pins=len(pins))))
