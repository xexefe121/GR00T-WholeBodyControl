$ErrorActionPreference='Stop'
$base=$PSScriptRoot
$folder=Join-Path $base 'argv_preflight_v2'
if(Test-Path -LiteralPath $folder){throw 'Preserve existing argument preflight.'}
[IO.Directory]::CreateDirectory($folder)|Out-Null
function Write-NewJson([string]$path,$value){$f=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$w=[IO.StreamWriter]::new($f,[Text.UTF8Encoding]::new($false));try{$w.Write(($value|ConvertTo-Json -Depth 12));$w.Flush()}finally{$w.Dispose()}}finally{$f.Dispose()}}
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $base 'run_recovery_durable_v2.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count -ne 0){throw 'Corrected launcher parse failed.'}
$assignment=$ast.Find({param($node) $node -is [Management.Automation.Language.AssignmentStatementAst] -and $node.Left -is [Management.Automation.Language.VariableExpressionAst] -and $node.Left.VariablePath.UserPath -eq 'arguments'},$true).Extent.Text
if(-not $assignment){throw 'Actual launcher argument expression missing.'}
$old=Get-Content -LiteralPath (Join-Path $base 'launch_receipt.json') -Raw|ConvertFrom-Json
$wanted=@($old.wsl_arguments)
$wanted[$wanted.Length-1]='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v1/argv_probe_v2.sh'
$wanted+=@('FIXED_ARG_ONE','FIXED_ARG_TWO')
$receipt=[pscustomobject]@{wsl_arguments=$wanted}
Invoke-Expression $assignment
$serialized=$arguments
$guard=@()
foreach($bad in @('contains space',('contains'+[char]34+'quote'),('contains'+[char]39+'quote'))){$receipt=[pscustomobject]@{wsl_arguments=@($bad)};$rejected=$false;try{Invoke-Expression $assignment}catch{$rejected=$true};if(-not $rejected){throw 'Unquoted-token guard failed.'};$guard+=@{value=$bad;rejected=$rejected}}
$started=[DateTime]::UtcNow.ToString('o')
$child=Start-Process -FilePath "$env:SystemRoot/System32/wsl.exe" -ArgumentList $serialized -WorkingDirectory $base -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'stdout.log') -RedirectStandardError (Join-Path $folder 'stderr.log')
$handle=$child.Handle;$childPid=$child.Id
$child.WaitForExit();$raw=$child.ExitCode
$child.Dispose()
$absent=($null -eq (Get-Process -Id $childPid -ErrorAction SilentlyContinue))
$output=$null
if($raw -eq 0){$output=Get-Content -LiteralPath (Join-Path $folder 'stdout.log') -Raw|ConvertFrom-Json}
$passed=$raw -eq 0 -and $absent -and $output.probe_only -eq $true -and $output.cwd -eq '/' -and $output.OMP_NUM_THREADS -eq '1' -and $output.OPENBLAS_NUM_THREADS -eq '1' -and $output.MKL_NUM_THREADS -eq '1' -and $output.PYTHONDONTWRITEBYTECODE -eq '1' -and $output.PYTHONPATH -eq ($wanted|Where-Object {$_ -like 'PYTHONPATH=*'}).Substring(11)
Write-NewJson (Join-Path $folder 'report.json') ([ordered]@{passed=$passed;started_utc=$started;ended_utc=[DateTime]::UtcNow.ToString('o');powershell_version=$PSVersionTable.PSVersion.ToString();child_pid=$childPid;handle_acquired=($handle -ne [IntPtr]::Zero);raw_exit_code=$raw;windows_child_absent=$absent;wsl_arguments=$wanted;serialized_arguments=$serialized;actual_launcher_assignment=$assignment;guard_tests=$guard;linux_output=$output;task_python_calls=0;model_calls=0;native_steps=0})
Get-Content -LiteralPath (Join-Path $folder 'report.json') -Raw
if(-not $passed){exit 1}
