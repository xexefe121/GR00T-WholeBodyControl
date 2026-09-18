$ErrorActionPreference = 'Stop'
$repo = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$out = 'E:\codex-artifacts\bfm_teleop_20260917\fix5\local_tcp_full_2'
$state = 'tcp://127.0.0.1:5568'
$target = 'tcp://127.0.0.1:5569'
$teleop = 'tcp://127.0.0.1:5570'
if (Test-Path $out) { throw "output exists: $out" }
New-Item -ItemType Directory -Path $out | Out-Null
$policy = Start-Process python -ArgumentList @('-m','gear_sonic.scripts.run_g1_true23_bfm_split_policy','run','--placement','windows','--state-endpoint',$state,'--target-endpoint',$target,'--teleop-endpoint',$teleop,'--output',(Join-Path $out 'policy'),'--duration-seconds','115.6','--priority','high','--torch-threads','4') -WorkingDirectory $repo -PassThru -RedirectStandardOutput (Join-Path $out 'policy.log') -RedirectStandardError (Join-Path $out 'policy.err')
$publisher = Start-Process python -ArgumentList @('-m','gear_sonic.scripts.run_g1_true23_bfm_teleop_sim','publish','--clip','pico','--endpoint',$teleop,'--start-delay','12','--data-root','C:\Users\camer\sonic23_sim_artifacts\internet_pico_20260909_v1') -WorkingDirectory $repo -PassThru -RedirectStandardOutput (Join-Path $out 'publisher.log') -RedirectStandardError (Join-Path $out 'publisher.err')
Start-Sleep -Seconds 12
try {
  $wslOutput = (wsl.exe -d Ubuntu-22.04 -- wslpath -u (Join-Path $out 'native')).Trim()
  wsl.exe -d Ubuntu-22.04 -- bash -lc "/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic_deploy/target/release/g1_true23_bfm_lowcmd_loop --source replay --replay /mnt/e/codex-artifacts/bfm_teleop_20260917/fix4/pico_lowstate_500hz.flatbin --initial-command /mnt/e/codex-artifacts/bfm_teleop_20260917/fix4/initial_command.bin --state-endpoint $state --target-endpoint $target --output $wslOutput --ticks 57800"
  $native = $LASTEXITCODE
} finally {
  $policy.WaitForExit(); $publisher.WaitForExit()
}
$policy.Refresh(); $publisher.Refresh()
@{native_exit=$native;policy_exit=$policy.ExitCode;publisher_exit=$publisher.ExitCode} | ConvertTo-Json | Set-Content (Join-Path $out 'process_status.json')
