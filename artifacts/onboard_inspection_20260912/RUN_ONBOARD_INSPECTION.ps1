param([ValidateSet('inventory', 'controller', 'factory')][string]$Stage = 'inventory')
$ErrorActionPreference = 'Stop'
$scriptPath = '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/onboard_inspection_20260912/run_readonly_ssh.py'
wsl.exe -d Ubuntu-22.04 --cd / -- python3 $scriptPath --stage $Stage
if ($LASTEXITCODE -ne 0) { throw 'Read-only robot inspection failed; no robot services or modes were changed.' }
