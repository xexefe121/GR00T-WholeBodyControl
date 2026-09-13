param([Parameter(Mandatory=$true)][string]$ReviewPath,[Parameter(Mandatory=$true)][string]$ReviewSha256)
$ErrorActionPreference='Stop'
$base='E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_full_state_secants_v1'
$process=Join-Path $base 'process'
$requestPath=Join-Path $base 'request.json'
function Read-TaskJson([string]$path){return ([IO.File]::ReadAllText($path) | ConvertFrom-Json)}
function Read-TaskHash([string]$path){
    $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    $hash=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose();$stream.Dispose()}
}
function Write-TaskJson([string]$path,$value){[IO.File]::WriteAllText($path,($value | ConvertTo-Json -Depth 100),[Text.UTF8Encoding]::new($false))}
function Check-TaskPins($pins){
    $files=[ordered]@{};$all=$true
    foreach($pin in $pins.PSObject.Properties){
        $actual=Read-TaskHash $pin.Name;$matched=($actual -ceq $pin.Value)
        $files[$pin.Name]=@{expected=$pin.Value;actual=$actual;matched=$matched}
        if(-not $matched){$all=$false}
    }
    return @{all_exact=$all;files=$files}
}
if((Read-TaskHash $ReviewPath) -cne $ReviewSha256){throw 'Final review identity changed.'}
$review=Read-TaskJson $ReviewPath
if($review.passed -ne $true -or $review.request_sha256 -cne (Read-TaskHash $requestPath) -or $review.launcher_sha256 -cne (Read-TaskHash $PSCommandPath)){throw 'Final root review binding invalid.'}
$request=Read-TaskJson $requestPath
if($request.generation_selected -ne $true -or $request.signed_rows -ne 354612 -or $request.rows -ne 3057 -or $request.axes -ne 58){throw 'Selected scope mismatch.'}
if(Test-Path -LiteralPath (Join-Path $base 'generation')){throw 'Generation output already exists.'}
foreach($name in @('start.json','stdout.log','stderr.log','exit.json','child.json')){if(Test-Path -LiteralPath (Join-Path $process $name)){throw 'Existing attempt must be preserved.'}}
$lock=[IO.File]::Open((Join-Path $process 'started.lock'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$lock.Dispose()
$start=@{wrapper_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');request_sha256=(Read-TaskHash $requestPath);review_path=$ReviewPath;review_sha256=$ReviewSha256;launcher_sha256=(Read-TaskHash $PSCommandPath)}
Write-TaskJson (Join-Path $process 'start.json') $start
$raw=-1;$exitCode=7;$errorText=$null;$child=$null;$childPid=$null;$postExact=$false
try{
    $pre=Check-TaskPins $request.windows_launch_sha256
    Write-TaskJson (Join-Path $process 'preflight_hashes.json') $pre
    if($pre.all_exact -ne $true){throw 'Launch inputs changed.'}
    $argsList='-d Ubuntu-22.04 --cd / -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_secants_v1/source_snapshot_v1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_secants_v1/source_snapshot_v1/generate_secants.py --request /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_secants_v1/request.json --output /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_secants_v1/generation'
    $child=Start-Process -FilePath 'wsl.exe' -ArgumentList $argsList -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $process 'stdout.log') -RedirectStandardError (Join-Path $process 'stderr.log')
    $nativeHandle=$child.Handle;$childPid=$child.Id
    Write-TaskJson (Join-Path $process 'child.json') @{child_pid=$childPid;wrapper_pid=$PID;handle_captured=$true;started_utc=[DateTime]::UtcNow.ToString('o')}
    $child.WaitForExit();$raw=$child.ExitCode
    if($null -eq $raw){$raw=-2;throw 'Raw child exit code unavailable.'}
    $exitCode=$raw
    if($raw -eq 0){
        $report=Read-TaskJson (Join-Path $base 'generation/report.json')
        if($report.complete -ne $true -or $report.request_sha256 -cne $start.request_sha256 -or $report.signed_rows -ne 354612 -or $report.overlap_rows -ne 140622 -or $report.new_rows -ne 213990 -or $report.all_inputs_unchanged -ne $true -or $report.all_runtime_and_sources_unchanged -ne $true -or $report.model_calls -ne 0 -or $report.physics_steps -ne 0){throw 'Completed producer verdict/count mismatch.'}
    }
}catch{$errorText=$_.Exception.Message;$exitCode=7}finally{
    try{
        $post=Check-TaskPins $request.windows_launch_sha256
        Write-TaskJson (Join-Path $process 'postrun_hashes.json') $post
        $postExact=$post.all_exact
        if(-not $postExact){$exitCode=7;$errorText='Postrun launch input hash mismatch.'}
        if((Read-TaskHash $requestPath) -cne $start.request_sha256 -or (Read-TaskHash $ReviewPath) -cne $ReviewSha256){$exitCode=7;$errorText='Request or final review changed.'}
    }catch{$exitCode=7;$errorText=$_.Exception.Message}
    Write-TaskJson (Join-Path $process 'exit.json') @{wrapper_pid=$PID;child_pid=$childPid;raw_python_exit_code=$raw;raw_exit_known=($null -ne $raw -and $raw -ge 0);exit_code=$exitCode;error=$errorText;all_postrun_hashes_exact=$postExact;ended_utc=[DateTime]::UtcNow.ToString('o');request_sha256=$start.request_sha256;review_sha256=$ReviewSha256}
    if($null -ne $child){$child.Dispose()}
}
exit $exitCode
