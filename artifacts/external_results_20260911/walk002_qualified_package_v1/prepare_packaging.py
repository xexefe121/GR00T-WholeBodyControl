"""Freeze reusable replay and rendering launchers; no dynamics or rendering."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

PACK=Path(__file__).parent
NEW=PACK.parent
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
QPATH=NEW/'walk002_hybrid_root_qualification_v1/qualification.json'
QSHA='a7eaec31dd572724921e2d2986ab8c6ab5db33008f6b3c1af5d2dbd72bf92e4b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def wsl(p):
    p=str(p).replace('\\','/');return '/mnt/'+p[0].lower()+p[2:]
assert sha(QPATH)==QSHA
q=json.loads(QPATH.read_text())
replay=PACK/'replay'
replay.mkdir(exist_ok=True)
for source,target in ((ROOT/'artifacts/teleop_resume_20260911/audit_restored_native_segment.py',replay/'audit_restored_native_segment.py'),
    (ROOT/'gear_sonic/utils/g1_true23_feasibility_referee.py',replay/'g1_true23_feasibility_referee.py')):
    if target.exists():assert target.read_bytes()==source.read_bytes()
    else:shutil.copyfile(source,target)
frozen=NEW/'walk002_terminal_bfm_hybrid_source_v1/repo'
for source in frozen.rglob('*'):
    if source.is_file() and '__pycache__' not in source.parts:
        target=replay/'repo'/source.relative_to(frozen);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==source.read_bytes()
        else:shutil.copyfile(source,target)

fixtures={'full':NEW/'walk002_canonical_initial_fixture_v1/initial_integration_state.npz',
          'hold':NEW/'walk002_hybrid_hold_independent_fixture_v1/initial_integration_state.npz'}
fixture_checks={}
for part,fixture in fixtures.items():
    with np.load(fixture,allow_pickle=False) as f,np.load(q['traces'][part]['path'],allow_pickle=False) as t:
        assert f['state_vector'].shape==(291,) and int(f['state_spec'])==8191
        assert f['state_vector'].tobytes()==t['initial_integration'].tobytes()
        fixture_checks[part]=dict(path=str(fixture),sha256=sha(fixture),all291_equals_qualified_segment_initial=True)
put(PACK/'replay_fixture_proof.json',fixture_checks)
pins={str(QPATH):QSHA}
for entry in [*q['traces'].values(),*q['independent_reports'].values()]:pins[entry['path']]=entry['sha256']
for p in [*fixtures.values(),*replay.rglob('*.py'),*replay.rglob('*.json')]:pins[str(p)]=sha(p)
bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
manifest=json.loads((bundle/'manifest.json').read_text())
for name in ('manifest.json','contract.json','native_prepared.xml','prepared_model_arrays.npz','walk002/native_original.npz','walk002/original29.npz','walk002/timeline.json'):
    p=bundle/name;pins[str(p)]=sha(p)
for name,digest in manifest['meshes'].items():
    p=bundle/'meshes'/name;assert sha(p)==digest;pins[str(p)]=digest
bootstrap=ROOT/'artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
pins[str(bootstrap)]=sha(bootstrap)
put(PACK/'replay_input_pins.json',pins)

launcher=PACK/'RUN_WALK002_RECORDED_PHYSICS_REPLAY.ps1'
text=(ROOT/'artifacts/teleop_resume_20260911/RUN_PICO_RECORDED_PHYSICS_REPLAY.ps1').read_text()
text=text.replace('PICO','WALK002 HYBRID').replace('pico_recorded_replay_','walk002_recorded_replay_')
text=text.replace("'pico_full_root_qualification_v1/qualification.json'","'walk002_hybrid_root_qualification_v1/qualification.json'")
text=text.replace('746502265523f527a7836f8b091ba521a56078629b1943468eff04f8bfa61daa',QSHA)
begin=text.index("Assert-Pin (Join-Path $repoWin 'artifacts/teleop_resume_20260911/audit_restored_native_segment.py')")
end=text.index('$qualification = ',begin)
text=text[:begin]+"Assert-Pin (Join-Path $baseWin 'walk002_qualified_package_v1/replay_input_pins.json') '"+sha(PACK/'replay_input_pins.json')+"'\n$inputPins = Get-Content -Raw -LiteralPath (Join-Path $baseWin 'walk002_qualified_package_v1/replay_input_pins.json') | ConvertFrom-Json\nforeach ($entry in $inputPins.PSObject.Properties) { Assert-Pin $entry.Name $entry.Value }\n"+text[end:]
text=text.replace('complete_offline_pico_pass','complete_offline_walk002_hybrid_pass')
begin=text.index('foreach ($entry in (Get-Content')
end=text.index('New-Item -ItemType Directory',begin)
text=text[:begin]+text[end:]
text=text.replace("'artifacts/teleop_resume_20260911/audit_restored_native_segment.py'",'"$baseWsl/walk002_qualified_package_v1/replay/audit_restored_native_segment.py"')
text=text.replace('"$baseWsl/preserved_walk_demo_v1/repo"','"$baseWsl/walk002_qualified_package_v1/replay/repo"')
text=text.replace("'gear_sonic/utils/g1_true23_feasibility_referee.py'",'"$baseWsl/walk002_qualified_package_v1/replay/g1_true23_feasibility_referee.py"')
text=text.replace("'--clip', 'pico'","'--clip', 'walk002'")
text=text.replace("'pico_full_control_lm_v1' 'pico_canonical_initial_fixture_v1' 6530","'walk002_terminal_bfm_hybrid_v1' 'walk002_canonical_initial_fixture_v1' 1417")
text=text.replace("'pico_terminal_hold_v1' 'pico_hold_independent_fixture_v1' 250","'walk002_terminal_bfm_hybrid_v1/post_lifecycle_hold_5s' 'walk002_hybrid_hold_independent_fixture_v1' 250")
text=text.replace('qualified_pico_recorded_command_native_physics_replay','qualified_walk002_hybrid_recorded_command_native_physics_replay')
text=text.replace('65300','14170').replace('67,800','16,670')
launcher.write_text(text)

for mode in ('render','replay'):
    process=PACK/(mode+'_process');process.mkdir(exist_ok=True)
    if mode=='render':
        command="& 'C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe' '"+str(PACK/'render_walk002_qualified.py')+"' --qualification '"+str(QPATH)+"' --qualification-sha256 '"+QSHA+"'"
        result=NEW/'walk002_qualified_hybrid_video_v1/render_receipt.json'
    else:
        command="& '"+str(launcher)+"' -RunName 'walk002_one_command_replay_smoke_v1'"
        result=NEW/'walk002_one_command_replay_smoke_v1/replay_result.json'
    extra=[PACK/'qualified_inputs.py',PACK/'render_walk002_qualified.py',PACK/'visual_grid_tests.xml',launcher,PACK/'replay_input_pins.json',PACK/'replay_fixture_proof.json']
    launchpins=dict(pins,**{str(p):sha(p) for p in extra})
    # Every child checks immutable qualification/source/assets before work.
    script="""$ErrorActionPreference = 'Stop'
$started = [DateTime]::UtcNow.ToString('o')
$exitStatus = 1
$errorMessage = $null
function Read-Sha([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $hash.Dispose(); $stream.Dispose() }
}
try {
    $receipt = Get-Content -Raw -LiteralPath 'RECEIPT' | ConvertFrom-Json
    foreach ($entry in $receipt.hashes.PSObject.Properties) { if ((Read-Sha $entry.Name) -ne $entry.Value) { throw ('Input changed: ' + $entry.Name) } }
    if (Test-Path -LiteralPath 'OUTPUTROOT') { throw 'Existing output; refusing duplicate execution' }
    @{state='running';pid=$PID;started_utc=$started} | ConvertTo-Json | Set-Content -LiteralPath 'RUNNING'
    COMMAND
    $exitStatus = $LASTEXITCODE
} catch { $errorMessage = $_.Exception.ToString(); Write-Error $errorMessage -ErrorAction Continue }
finally {
    $result = @{pid=$PID;started_utc=$started;finished_utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitStatus;error=$errorMessage;result_exists=(Test-Path -LiteralPath 'RESULT')}
    if (-not $result.result_exists) { $exitStatus=1; $result.exit_code=1 }
    if ($result.result_exists) { $result.result_sha256=Read-Sha 'RESULT' }
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath 'STATUS'
}
exit $exitStatus
""".replace('RECEIPT',str(process/'launch_receipt.json')).replace('OUTPUTROOT',str(result.parent)).replace('RUNNING',str(process/'running.json')).replace('COMMAND',command).replace('RESULT',str(result)).replace('STATUS',str(process/'exit_status.json'))
    scriptpath=process/'run_durable.ps1';scriptpath.write_text(script)
    launchpins[str(scriptpath)]=sha(scriptpath)
    put(process/'launch_receipt.json',dict(kind='qualified_'+mode,command=command,hashes=launchpins,qualification_sha256=QSHA,
        no_controller_inference=True,render_no_dynamics=mode=='render',replay_native_steps=16670 if mode=='replay' else 0))
print(json.dumps(dict(replay_launcher=str(launcher),replay_launcher_sha256=sha(launcher),pins=len(pins))))
