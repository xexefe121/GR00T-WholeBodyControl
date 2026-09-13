"""Run one declared independent audit with saved logs and exact input receipts."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
BOOT='/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def utc():return datetime.now(timezone.utc).isoformat()
def windows(path):
 if path.startswith('/mnt/') and path[6:7]=='/':return Path(path[5].upper()+':'+path[6:])
 return Path(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--commands',type=Path,required=True)
 ap.add_argument('--commands-sha256',required=True);ap.add_argument('--stage',required=True);a=ap.parse_args()
 assert sha(a.commands)==a.commands_sha256
 q=read(a.commands);assert q['preparation_pass'] is True
 selected=[stage for stage in q['commands'] if stage['stage']==a.stage];assert len(selected)==1
 stage=selected[0];args=stage['arguments'];target=windows(args[args.index('--output')+1])
 assert not target.exists(),'Preserve previous audit output'
 pins=dict(q['input_sha256']);pins[a.commands.resolve().as_posix()]=sha(a.commands)
 pins[Path(__file__).resolve().as_posix()]=sha(__file__)
 for name in ('--physical-audit','--fixture','--matching-trace','--preceding-trace'):
  if name in args:
   path=windows(args[args.index(name)+1]);pins[path.as_posix()]=sha(path)
 for path,h in pins.items():assert sha(path)==h,path
 folder=HERE/'actual_audit_processes'/a.stage;folder.mkdir(parents=True,exist_ok=False)
 command=['C:/Windows/System32/wsl.exe','-d','Ubuntu-22.04','--cd','/','--','bash',BOOT,'env']
 command.extend(k+'='+v for k,v in q['environment'].items());command.extend([q['python'],'-B',*args])
 save(folder/'start.json',dict(started_utc=utc(),exact_argv=command,input_sha256=pins,
  requested_native_step_max=stage['native_step_max'],model_calls=0,automatic_retry=False))
 with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
  process=subprocess.Popen(command,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  save(folder/'child.json',dict(pid=process.pid,handle_acquired=True,started_utc=utc()))
  code=process.wait()
 post={path:dict(expected=h,actual=sha(path)) for path,h in pins.items()}
 exact=all(v['expected']==v['actual'] for v in post.values())
 save(folder/'exit.json',dict(completed_utc=utc(),exit_code=code,raw_exit_known=True,child_reaped=True,
  all_postrun_pins_exact=exact,input_checks=post,output_report_sha256=sha(target/'report.json') if (target/'report.json').exists() else None))
 print(json.dumps(dict(stage=a.stage,exit_code=code,all_postrun_pins_exact=exact,report=str(target/'report.json'),process_receipts=str(folder))))
 if not exact:raise RuntimeError('Audit input changed')
 return code
if __name__=='__main__':sys.exit(main())
