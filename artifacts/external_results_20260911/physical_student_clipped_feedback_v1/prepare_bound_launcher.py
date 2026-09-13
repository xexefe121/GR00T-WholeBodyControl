"""Generate one durable launcher only after concrete final binding passes."""
import argparse
import json
from pathlib import Path
import sys

BASE=Path(__file__).parent;sys.path.insert(0,str(BASE/'source_snapshot_v1'))
from evaluation_gate import read,sha,require_ready
from head_activation_witness import require_witness_ready


def quote(value):return "'"+str(value).replace("'","''")+"'"
def write(path,value):Path(path).write_text(value,encoding='utf-8')
HASH="""$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
"""


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['evaluation'],required=True)
    args=parser.parse_args();mode=args.mode
    if mode=='witness':require_witness_ready(BASE)
    else:require_ready(BASE)
    binding_path=BASE/(mode+'_binding.json');binding=read(binding_path)
    folder=BASE/(mode+'_process');folder.mkdir(exist_ok=False)
    exit_test=BASE.parent/'broader_labels_independent_v1/exit_capture_test.json'
    check=read(exit_test);assert check['actual_exit']==check['expected_exit']==7 and check['handle_acquired_before_wait']
    linux='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/'+BASE.name
    script='head_activation_witness.py' if mode=='witness' else 'evaluate_physical_response_student.py'
    bootstrap='Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
    arguments=['-d','Ubuntu-22.04','--cd','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof','--','bash',
        '/mnt/z'+bootstrap[2:],'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'PYTHONPATH='+linux+'/source_snapshot_v1','/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',linux+'/source_snapshot_v1/'+script]
    launcher=folder/'run.ps1';durable=folder/'run_durable.ps1'
    # Duplicate-output guards apply before starting the actual WSL command.
    forbidden=['head_witness'] if mode=='witness' else ['nominal','post_lifecycle_hold_5s','pilot_outcome.json']
    no_outputs=''.join("if (Test-Path -LiteralPath "+quote(BASE/name)+") { throw 'Existing attempt output must be preserved.' }\n" for name in forbidden)
    no_outputs+="$executionLock = [System.IO.File]::Open("+quote(folder/'execution.lock')+",[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)\n$executionLock.Close()\n"
    guard=HASH+'$bindingPath = '+quote(binding_path)+'\n'+\
        'if ((Get-TaskHash $bindingPath) -ne '+quote(sha(binding_path))+') { throw \'Final binding changed.\' }\n'+\
        '$binding = Get-Content -Raw -LiteralPath $bindingPath | ConvertFrom-Json\n'+\
        "foreach ($entry in $binding.input_files) { if ((Get-TaskHash $entry.path) -ne $entry.sha256) { throw ('Bound input changed: ' + $entry.path) } }\n"
    command='$arguments = @(\n'+',\n'.join('    '+quote(a) for a in arguments)+'\n)\n'+\
        '& "$env:SystemRoot\\System32\\wsl.exe" @arguments\n$taskExit = $LASTEXITCODE\n'+\
        "if ($null -eq $taskExit) { throw 'WSL exit status unavailable.' }\nif ($taskExit -ne 0) { exit $taskExit }\n"
    if mode=='witness':
        verdict='$result = Get-Content -Raw -LiteralPath '+quote(BASE/'head_witness/report.json')+' | ConvertFrom-Json\n'+\
            "if (-not $result.pass_all -or $result.attempted_head_calls -ne 1 -or $result.expected_head_calls -ne 1 -or $result.BFM_inference_calls -ne 0 -or $result.physics_steps -ne 0 -or $result.fitting_launched) { throw 'Single-call witness incomplete.' }\n"
    else:
        verdict='$result = Get-Content -Raw -LiteralPath '+quote(BASE/'pilot_outcome.json')+' | ConvertFrom-Json\n'+\
            "if (-not $result.nominal.full_segment_completed -or -not $result.extension.full_segment_completed -or $result.nominal.completed_controls -ne 1569 -or $result.extension.completed_controls -ne 250 -or $result.nominal.physics_steps -ne 15690 -or $result.extension.physics_steps -ne 2500) { throw 'Canonical evaluation stopped or incomplete; preserve first failure.' }\n"+\
            "if (-not $result.nominal.quiet_standing_diagnostic.quiet_standing_diagnostic_pass -or -not $result.extension.quiet_standing_diagnostic.quiet_standing_diagnostic_pass) { throw 'Physical completion retained; quiet standing failed.' }\n"
        for name in ('canonical_prefix250_parity.json','actual_query250_input_parity.json','actual_query250_ownexport_output_parity.json','original_prefix265_and_first_feedback_parity.json','first_feedback_actor_input265_parity.json','first_feedback_lag_history266_parity.json','first_feedback_actor_lag266_parity.json'):
            verdict+='$parity = Get-Content -Raw -LiteralPath '+quote(BASE/name)+" | ConvertFrom-Json\nif (-not $parity.passed) { throw 'Canonical transition parity failed.' }\n"
    write(launcher,guard+no_outputs+command+verdict+'exit 0\n')
    write(durable,HASH+'$folder = '+quote(folder)+'\n'+
        "$receiptPath = Join-Path $folder 'launch_receipt.json'\n$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json\n"+
        "foreach ($entry in $receipt.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Launch input changed: ' + $entry.Name) } }\n"+
        "$lock = [System.IO.File]::Open((Join-Path $folder 'started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)\n$lock.Close()\n"+
        "[ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;receipt_sha256=(Get-TaskHash $receiptPath)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'start.json') -Encoding UTF8\n"+
        "$exitCode = 1\n$errorText = $null\ntry {\n"+
        "    $taskChild = Start-Process -FilePath \"$env:SystemRoot\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'stdout.log') -RedirectStandardError (Join-Path $folder 'stderr.log')\n"+
        "    $nativeHandle = $taskChild.Handle\n"+
        "    [ordered]@{child_pid=$taskChild.Id;wrapper_pid=$PID;handle_acquired=($nativeHandle -ne [IntPtr]::Zero)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'child.json') -Encoding UTF8\n"+
        "    $taskChild.WaitForExit()\n    $exitCode = $taskChild.ExitCode\n    if ($null -eq $exitCode) { throw 'Child exit status unavailable.' }\n"+
        "    $post = [ordered]@{}\n    foreach ($entry in $receipt.input_hashes.PSObject.Properties) { $actual = Get-TaskHash $entry.Name; $post[$entry.Name] = $actual; if ($actual -ne $entry.Value) { throw ('Postrun input changed: ' + $entry.Name) } }\n"+
        "    $post | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $folder 'postrun_hashes.json') -Encoding UTF8\n"+
        "} catch { $errorText=$_.Exception.Message; $exitCode=1 } finally {\n"+
        "    [ordered]@{utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitCode;error=$errorText} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'exit.json') -Encoding UTF8\n}\nexit $exitCode\n")
    pins={entry['path']:entry['sha256'] for entry in binding['input_files']}
    for path in (binding_path,Path(__file__),launcher,durable,exit_test,Path(bootstrap)):
        pins[str(path).replace('\\','/')]=sha(path)
    receipt=dict(kind='one_selected_clipped_feedback_final75000_'+mode,binding_sha256=sha(binding_path),exact_wsl_arguments=arguments,input_hashes=pins,
        requested_main_controls=0 if mode=='witness' else 1569,conditional_hold_controls=0 if mode=='witness' else 250,
        expected_separate_head_calls=1 if mode=='witness' else 0,preparation_only=True,hardware_authorized=False)
    write(folder/'launch_receipt.json',json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(mode=mode,launch_receipt_sha256=sha(folder/'launch_receipt.json'),pins=len(pins),launched=False)))


if __name__=='__main__':main()
