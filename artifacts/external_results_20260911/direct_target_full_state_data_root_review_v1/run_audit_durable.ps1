$ErrorActionPreference='Stop'
$taskBase='E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_full_state_data_root_review_v1'
$taskProducer='E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_full_state_secants_v1'
function TaskHash([string]$path) {
    $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    $algorithm=[Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $algorithm.Dispose();$stream.Dispose() }
}
function TaskJson([string]$path,$value) { [IO.File]::WriteAllText($path,($value|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($false)) }
$pins=[ordered]@{
    "$taskBase\audit_saved_data.py"='94b9281198f1b62d6b55f573ab11e5d5328093a3bb4869e2e6977f014f55f1f8'
    "$taskBase\saved_math.py"='df8f1b15d9d83dca92666b6a44d33dd921fdc913f41795cd60ccd6377e53ad6b'
    "$taskProducer\request.json"='474d5e4a04f84702b8ada696560df4baf6a3ebfa3b0ffebb0c343d4da11ebde1'
    'E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_full_state_auditor_pico_review_v2\review.json'='a6ed5dc0cdd757fb06792539435feea4ff54b233c8d9ba47c9b05451c82e42e2'
    'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\teleop_six_hour_20260910\RUN_PASSING_WALK_WSL.sh'='392de6eccb281c41219566c4b7c9c1813b086913f66d861529d4904959168ebc'
}
foreach($entry in $pins.GetEnumerator()) { if((TaskHash $entry.Key) -cne $entry.Value) { throw ('Audit input changed: '+$entry.Key) } }
if(Test-Path -LiteralPath "$taskBase\results_v1") { throw 'Preserve existing actual audit results.' }
$producerReport=([IO.File]::ReadAllText("$taskProducer\generation\report.json")|ConvertFrom-Json)
$producerExit=([IO.File]::ReadAllText("$taskProducer\process\exit.json")|ConvertFrom-Json)
if($producerReport.complete -ne $true -or $producerExit.exit_code -ne 0 -or $producerExit.all_postrun_hashes_exact -ne $true) { throw 'Producer not successfully completed.' }
$processDir="$taskBase\process_v1"
New-Item -ItemType Directory -Path $processDir -ErrorAction Stop | Out-Null
TaskJson "$processDir\start.json" @{wrapper_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');launcher_sha256=(TaskHash $PSCommandPath);pins=$pins;producer_report_sha256=(TaskHash "$taskProducer\generation\report.json");producer_exit_sha256=(TaskHash "$taskProducer\process\exit.json")}
$child=$null;$childPid=$null;$raw=-1;$code=7;$errorText=$null;$postExact=$false
try {
    $taskArgs='-d Ubuntu-22.04 --cd / -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_data_root_review_v1/audit_saved_data.py --request /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_secants_v1/request.json --generation /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_secants_v1/generation --training-request /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_continuation_v1/training_request.json --original-feature-source /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_fp64_export_evaluation_v2/source_draft_v1/direct_features.py --output /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_data_root_review_v1/results_v1'
    $child=Start-Process -FilePath 'wsl.exe' -ArgumentList $taskArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput "$processDir\stdout.log" -RedirectStandardError "$processDir\stderr.log"
    $nativeHandle=$child.Handle;$childPid=$child.Id
    TaskJson "$processDir\child.json" @{wrapper_pid=$PID;child_pid=$childPid;handle_captured=$true}
    $child.WaitForExit();$raw=$child.ExitCode
    if($null -eq $raw) { $raw=-2;throw 'Unknown child exit.' }
    $code=$raw
    if($raw -eq 0) {
        $report=([IO.File]::ReadAllText("$taskBase\results_v1\report.json")|ConvertFrom-Json)
        if($report.data_review_pass -ne $true -or $report.rows_checked -ne 354612 -or $report.independently_recomputed_new_features_and_maps -ne 213990 -or $report.incompatible_normalized_float32_target_rows -ne 0) { throw 'Saved audit verdict/count mismatch.' }
    }
} catch { $code=7;$errorText=$_.Exception.Message }
finally {
    try {
        $postExact=$true
        foreach($entry in $pins.GetEnumerator()) { if((TaskHash $entry.Key) -cne $entry.Value) { $postExact=$false;throw ('Postrun pin changed: '+$entry.Key) } }
    } catch { $code=7;$errorText=$_.Exception.Message }
    TaskJson "$processDir\exit.json" @{wrapper_pid=$PID;child_pid=$childPid;raw_python_exit_code=$raw;exit_code=$code;error=$errorText;all_postrun_pins_exact=$postExact;ended_utc=[DateTime]::UtcNow.ToString('o')}
    if($null -ne $child) { $child.Dispose() }
}
exit $code
