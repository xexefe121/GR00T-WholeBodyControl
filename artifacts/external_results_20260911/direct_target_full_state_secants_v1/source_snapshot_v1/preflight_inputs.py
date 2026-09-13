"""Actual schemas and runtime pins only; zero task feature/map/probe calls."""
import argparse,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import scipy
from direct_features import DirectFeatures  # import only; never construct a task builder
from input_schema import validate_inputs
from secant_math import cell_ids

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def local(value):
    s=str(value).replace('\\','/')
    if len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+s[2:]
    return Path(s)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def run(config_path,dest):
    if sys.platform=='win32' or np.__version__!='1.26.4':raise ValueError('pinned preflight runtime')
    config=read(config_path);paths={k:local(v) for k,v in config['paths'].items()}
    loaded,schemas=validate_inputs(paths,read(paths['contract']))
    c=loaded['centers'];_,cell=cell_ids(c['dataset'],c['control'])
    margin=np.minimum(c['qpos'][:,7:]-c['joint_limits'][:,0],c['joint_limits'][:,1]-c['qpos'][:,7:])
    if not np.all(margin>.01):raise ValueError('fixed joint margin')
    producer=read(paths['generation_report']);review=read(paths['generation_review']);audit=read(paths['generation_audit'])
    if producer['complete'] is not True or review['passed'] is not True or audit['passed'] is not True:raise ValueError('existing data qualification')
    if audit['generation_report_sha256']!=sha(paths['generation_report']):raise ValueError('existing producer/audit identity')
    for role in ('centers','velocity_features','velocity_target','velocity_raw','velocity_feedback_clipped','velocity_native_clipped','velocity_value'):
        if producer['output_sha256'][paths[role].name]!=sha(paths[role]):raise ValueError('qualified output hash '+role)
    runtime=set()
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and Path(filename).is_file():runtime.add(Path(filename).resolve())
    for package in (Path(np.__file__).parent,Path(scipy.__file__).parent):
        for path in package.rglob('*'):
            if path.is_file() and (path.suffix=='.py' or '.so' in path.name):runtime.add(path.resolve())
        libs=package.with_name(package.name+'.libs')
        if libs.exists():runtime.update(p.resolve() for p in libs.rglob('*') if p.is_file())
    runtime.add(Path(sys.executable).resolve())
    result={'passed':True,'config_sha256':sha(config_path),'archive_schemas':schemas,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'input_sha256':{str(path):sha(path) for path in paths.values()},'runtime_sha256':{str(path):sha(path) for path in sorted(runtime)},
        'cell_centers':np.bincount(cell,minlength=9).tolist(),'joint_margin_min':float(margin.min()),
        'task_feature_calls':0,'task_map_calls':0,'task_probe_rows':0,'model_calls':0,'physics_steps':0}
    Path(dest).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'preflight_sha256':sha(dest),'runtime_pins':len(runtime),'inputs':len(paths),'versions':result['versions']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.config),Path(a.output))
