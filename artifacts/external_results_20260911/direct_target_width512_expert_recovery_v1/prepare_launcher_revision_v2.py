"""Launcher-only correction; immutable task sources/request/frozen inputs unchanged."""
import hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new(name,text):
    with (BASE/name).open('x',newline='\n') as f:f.write(text)
preserved=BASE/'launch_preserved_v1';preserved.mkdir(exist_ok=False)
for name in ('execution_clearance.json','owner_completion.json'):
    (preserved/name).write_bytes((BASE/name).read_bytes())
source=(BASE/'run_recovery_durable.ps1').read_text()
source=source.replace("'recovery_process_v1'","'recovery_process_v2'").replace("'launch_receipt.json'","'launch_receipt_v2.json'")
old='    $arguments=($receipt.wsl_arguments | ForEach-Object {if($_ -match \'["\\r\\n]\'){throw \'Invalid fixed WSL argument.\'};\'"\'+$_+\'"\'}) -join \' \''
fixed="    $arguments=($receipt.wsl_arguments | ForEach-Object {if(($_ -match '\\s') -or $_.IndexOf([char]34) -ge 0 -or $_.IndexOf([char]39) -ge 0){throw 'Invalid fixed WSL argument.'};$_}) -join ' '"
assert old in source;source=source.replace(old,fixed)
new('run_recovery_durable_v2.ps1',source)
new('run_recovery_v2.sh',(BASE/'run_recovery.sh').read_text().replace('recovery_process_v1','recovery_process_v2'))
owner=(BASE/'verify_recovery_completed.py').read_text().replace("'recovery_process_v1'","'recovery_process_v2'")
owner=owner.replace("'launch_receipt.json'","'launch_receipt_v2.json'").replace("'run_recovery_durable.ps1'","'run_recovery_durable_v2.ps1'")
owner=owner.replace("'owner_completion.json'","'owner_completion_v2.json'")
new('verify_recovery_completed_v2.py',owner)
print('Prepared launcher v2; no dispatch or task calls.')
