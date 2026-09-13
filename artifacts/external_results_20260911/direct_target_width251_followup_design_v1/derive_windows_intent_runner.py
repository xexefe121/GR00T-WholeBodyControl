"""Preserve native audit runner and derive Windows-only saved intent execution."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
src=HERE/'run_prepared_audit.py'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='ce6f78c6ebae5421fc6140de34dac6b98991522e7c653ce1da4b2201813c18e7'
text=src.read_text()
def change(old,new):
 global text
 assert text.count(old)==1,(old,text.count(old))
 text=text.replace(old,new)
change('import argparse,hashlib,json,subprocess,sys','import argparse,hashlib,json,subprocess,sys,os')
change("stage=selected[0];args=stage['arguments'];target=windows(args[args.index('--output')+1])",
 "stage=selected[0];args=stage['arguments'];target=windows(args[args.index('--output')+1])\n assert a.stage in ('main_intent','hold_intent') and stage['native_step_max']==0\n assert Path(args[0]).name=='inspect_recorded_intent.py'\n assert sys.version_info[:2]==(3,10)")
change("folder=HERE/'actual_audit_processes'/a.stage;folder.mkdir(parents=True,exist_ok=False)",
 "folder=HERE/'actual_audit_processes'/(a.stage+'_windows_v2');folder.mkdir(parents=True,exist_ok=False)")
change("command=['C:/Windows/System32/wsl.exe','-d','Ubuntu-22.04','--cd','/','--','bash',BOOT,'env']\n command.extend(k+'='+v for k,v in q['environment'].items());command.extend([q['python'],'-B',*args])",
 "add_pin(pins,sys.executable)\n command=[sys.executable,'-B',*[str(windows(v)) if v.startswith('/mnt/') else v for v in args]]\n environment=os.environ.copy()\n environment.update({k:str(windows(v)) if v.startswith('/mnt/') else v for k,v in q['environment'].items()})\n environment['PYTHONDONTWRITEBYTECODE']='1'")
change('creationflags=subprocess.CREATE_NO_WINDOW)','creationflags=subprocess.CREATE_NO_WINDOW,env=environment)')
with (HERE/'run_windows_intent.py').open('x',encoding='utf-8') as f:f.write(text)
print('Derived Windows saved-intent runner; no audit execution.')
