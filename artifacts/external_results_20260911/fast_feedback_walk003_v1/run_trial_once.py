"""Capture one fresh WSL rollout and its raw exit; no automatic retry."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,hashlib
BASE=Path(__file__).resolve().parent
process=BASE/'baseline_process_v1';process.mkdir(exist_ok=False)
output=BASE/'baseline_v1';assert not output.exists()
def posix(p):
    s=Path(p).resolve().as_posix();return '/mnt/'+s[0].lower()+s[2:]
def write(p,r):
    with p.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2);f.write('\n')
boot='/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
command=['C:/Windows/System32/wsl.exe','-d','Ubuntu-22.04','--cd','/','--','bash',boot,'env',
    'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
    'PYTHONPATH=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python','-B','-u',posix(BASE/'run_controller.py'),'--output',posix(output)]
write(process/'start.json',dict(utc=datetime.now(timezone.utc).isoformat(),command=command,
    source_sha256=hashlib.sha256((BASE/'run_controller.py').read_bytes()).hexdigest(),automatic_retry=False))
with (process/'stdout.log').open('xb') as out,(process/'stderr.log').open('xb') as err:
    child=subprocess.Popen(command,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    write(process/'child.json',dict(pid=child.pid));code=child.wait()
write(process/'exit.json',dict(utc=datetime.now(timezone.utc).isoformat(),raw_exit_code=code,child_reaped=True,automatic_retry=False))
print(json.dumps(dict(raw_exit_code=code,output=str(output))))
sys.exit(code)
