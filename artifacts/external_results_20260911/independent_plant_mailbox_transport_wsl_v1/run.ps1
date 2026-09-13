$ErrorActionPreference = 'Stop'
$folder = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\independent_plant_mailbox_transport_wsl_v1'
function Write-NewJson([string]$name, $value) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($value | ConvertTo-Json -Depth 12))
    $stream = [System.IO.File]::Open((Join-Path $folder $name),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    try { $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
}
function Hash([string]$path) {
    $alg=[System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($alg.ComputeHash([System.IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant() } finally { $alg.Dispose() }
}
$arguments = @('-d','Ubuntu-22.04','--cd','/','--','bash',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python','-B',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/independent_plant_mailbox_transport_wsl_v1/run_and_account.py')
$requestPath=Join-Path $folder 'request.json'
if ((Hash $requestPath) -ne '1cbcb6b6349bf9cf052a3f8662dfc021a84ca6ce89b27c89cbe95036f1900399') { throw 'Selected request changed' }
Write-NewJson 'launch_receipt.json' @{arguments=$arguments;wrapper_pid=$PID;request_sha256=(Hash $requestPath);launcher_sha256=(Hash $PSCommandPath);bootstrap_sha256=(Hash 'Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh');start_utc=[DateTime]::UtcNow.ToString('o')}
$code=$null;$failure=$null;$child=$null
try {
    $child=Start-Process -FilePath 'C:\Windows\System32\wsl.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'wsl_stdout.log') -RedirectStandardError (Join-Path $folder 'wsl_stderr.log')
    $handle=$child.Handle
    Write-NewJson 'windows_child.json' @{wrapper_pid=$PID;child_pid=$child.Id;handle_acquired=$true}
    $child.WaitForExit();$code=$child.ExitCode
} catch { $failure=$_.Exception.ToString() }
Write-NewJson 'windows_exit.json' @{raw_exit_code=$code;raw_exit_known=($null -ne $code);exception=$failure;end_utc=[DateTime]::UtcNow.ToString('o');request_sha256=(Hash $requestPath)}
if ($null -eq $code -or $null -ne $failure) { exit 1 }
exit $code
