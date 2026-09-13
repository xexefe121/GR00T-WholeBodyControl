"""Run one declared independent audit with saved logs and exact input receipts."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
BOOT='/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
BOOT_SHA256='392de6eccb281c41219566c4b7c9c1813b086913f66d861529d4904959168ebc'
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
def canonical(path):return windows(str(path).replace('\\','/')).resolve().as_posix().casefold()

def add_pin(pins,path,expected=None):
 path=windows(str(path).replace('\\','/')).resolve();digest=sha(path)
 if expected is not None and digest!=expected:raise ValueError('Prepared input changed: '+str(path))
 aliases=[(p,h) for p,h in pins.items() if canonical(p)==canonical(path)]
 if any(h!=digest for _,h in aliases):raise ValueError('Conflicting or changed input alias: '+str(path))
 if not aliases:pins[path.as_posix()]=digest
 return digest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--commands',type=Path,required=True)
 ap.add_argument('--commands-sha256',required=True);ap.add_argument('--stage',required=True);a=ap.parse_args()
 assert sha(a.commands)==a.commands_sha256
 q=read(a.commands);assert q['preparation_pass'] is True
 selected=[stage for stage in q['commands'] if stage['stage']==a.stage];assert len(selected)==1
 stage=selected[0];args=stage['arguments'];target=windows(args[args.index('--output')+1])
 assert not target.exists(),'Preserve previous audit output'
 pins={}
 for path,digest in q['input_sha256'].items():add_pin(pins,path,digest)
 add_pin(pins,a.commands,a.commands_sha256);add_pin(pins,Path(__file__))
 add_pin(pins,windows(BOOT),BOOT_SHA256)
 for name in ('--physical-audit','--fixture','--matching-trace','--preceding-trace'):
  if name in args:
   path=windows(args[args.index(name)+1]);add_pin(pins,path)
 for path,h in pins.items():assert sha(path)==h,path
 folder=HERE/'actual_audit_processes'/a.stage;folder.mkdir(parents=True,exist_ok=False)
 command=['C:/Windows/System32/wsl.exe','-d','Ubuntu-22.04','--cd','/','--','bash',BOOT,'env']
 command.extend(k+'='+v for k,v in q['environment'].items());command.extend([q['python'],'-B',*args])
 save(folder/'start.json',dict(started_utc=utc(),exact_argv=command,input_sha256=pins,
  requested_native_step_max=stage['native_step_max'],model_calls=0,automatic_retry=False))
 process=None;code=None;raw_known=False;child_reaped=False;run_error=None;raised=None
 try:
  with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
   process=subprocess.Popen(command,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
   save(folder/'child.json',dict(pid=process.pid,handle_acquired=True,started_utc=utc()))
   code=process.wait()
   if type(code) is not int:raise RuntimeError('Child wait returned no known integer exit')
   raw_known=True;child_reaped=True
 except BaseException as exc:
  raised=exc;run_error=dict(type=type(exc).__name__,message=str(exc))
 post={}
 for path,h in pins.items():
  actual=None;error=None
  try:actual=sha(path)
  except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc))
  post[path]=dict(expected=h,actual=actual,error=error)
 exact=all(v['error'] is None and v['expected']==v['actual'] for v in post.values())
 report_hash=None;report_error=None
 try:
  if (target/'report.json').exists():report_hash=sha(target/'report.json')
 except BaseException as exc:report_error=dict(type=type(exc).__name__,message=str(exc))
 accounting=raw_known and child_reaped and run_error is None and exact and report_error is None
 save(folder/'exit.json',dict(completed_utc=utc(),exit_code=code,raw_child_exit_code=code,
  raw_exit_known=raw_known,child_reaped=child_reaped,child_pid=None if process is None else process.pid,
  process_error=run_error,accounting_passed=accounting,
  all_postrun_pins_exact=exact,input_checks=post,output_report_sha256=report_hash,output_report_hash_error=report_error))
 print(json.dumps(dict(stage=a.stage,exit_code=code,all_postrun_pins_exact=exact,report=str(target/'report.json'),process_receipts=str(folder))))
 if raised is not None:raise raised
 if not accounting:raise RuntimeError('Audit preservation failed; inspect saved exit and per-file errors')
 return code
if __name__=='__main__':sys.exit(main())
