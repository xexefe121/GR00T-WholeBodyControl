"""Pause only this pilot during a timing test; resume it in a finally block."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--braking',action='store_true')
p.add_argument('--realtime-priority',action='store_true')
p.add_argument('--initial-velocity',type=float,default=0.)
p.add_argument('--fault-control',type=int)
p.add_argument('--spin-us',type=int,default=0);a=p.parse_args()
expected={('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/'+name).encode() for name in ('pilot_v1','pilot_balanced_v1')}
matches=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:args=(entry/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    if b'gear_sonic.scripts.train_g1_true23_causal_dynamics' in args and expected.intersection(args):matches.append(int(entry.name))
if len(matches)>1:raise RuntimeError('ambiguous pilot process identity')
started=time.time();paused=[]
try:
    for pid in matches:os.kill(pid,signal.SIGSTOP);paused.append(pid)
    time.sleep(.5)
    command=[sys.executable,str(Path(__file__).with_name('run_native_clock.py')),'--optimize','--output',str(a.output),'--initial-velocity',str(a.initial_velocity)]
    if a.braking:command.append('--braking')
    if a.fault_control is not None:command+=['--fault-control',str(a.fault_control)]
    command+=['--spin-us',str(a.spin_us)]
    if a.realtime_priority:command.append('--realtime-priority')
    subprocess.run(command,check=True)
finally:
    resumed=[]
    for pid in paused:
        try:os.kill(pid,signal.SIGCONT);resumed.append(pid)
        except ProcessLookupError:pass
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'training_isolation.json').write_text(json.dumps(dict(paused_pids=paused,resumed_pids=resumed,elapsed_s=time.time()-started),indent=2)+'\n')
