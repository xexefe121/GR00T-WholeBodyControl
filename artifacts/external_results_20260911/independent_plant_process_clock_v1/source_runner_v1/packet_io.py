"""Small lossless postrun writer and concrete request gates; no native imports."""
import base64
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
import numpy as np


def local(value):
    value=str(value).replace('\\','/')
    if sys.platform!='win32' and len(value)>2 and value[1]==':':
        value='/mnt/'+value[0].lower()+value[2:]
    return Path(value)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def lossless(value):
    """No diagnostic truncation, mutable aliases, NaN coercion or object pickle."""
    if value is None or type(value) in (bool,int,str):return value
    if type(value) is float:return {'type':'float','hex':value.hex()}
    if type(value) is bytes:
        return {'type':'bytes','base64':base64.b64encode(value).decode('ascii'),
                'sha256':hashlib.sha256(value).hexdigest(),'length':len(value)}
    if dataclasses.is_dataclass(value):
        return {'type':'dataclass','class':type(value).__name__,
                'fields':{f.name:lossless(getattr(value,f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value,Mapping):
        if any(type(k) is not str for k in value):raise TypeError('string evidence keys required')
        return {k:lossless(v) for k,v in value.items()}
    if type(value) in (list,tuple):return [lossless(v) for v in value]
    raise TypeError('unsupported owned evidence: '+type(value).__name__)


def save_owned(output,session):
    from evidence import export_owned
    output=Path(output);arrays,metadata,capsules=export_owned(session)
    with (output/'trace.npz').open('xb') as stream:np.savez(stream,**arrays)
    write(output/'evidence.json',lossless(metadata))
    folder=output/'raw_capsules';folder.mkdir(exist_ok=False);index={}
    for name,value in capsules.items():
        if type(value) is not bytes:raise TypeError('raw capsule must remain exact bytes: '+name)
        path=folder/(name+'.bin')
        with path.open('xb') as stream:stream.write(value)
        index[name]={'path':path.name,'bytes':len(value),'sha256':sha(path)}
    write(folder/'index.json',index)
    return {'arrays':{k:{'shape':list(v.shape),'dtype':v.dtype.str} for k,v in arrays.items()},
            'trace_sha256':sha(output/'trace.npz'),'evidence_sha256':sha(output/'evidence.json'),
            'raw_capsule_index_sha256':sha(folder/'index.json')}


def pin_check(entries):
    files={}
    for entry in entries:
        path=local(entry['path']).resolve()
        if path in files:raise ValueError('duplicate resolved input pin')
        actual=sha(path)
        files[path]={'expected':entry['sha256'],'actual':actual,'matched':actual==entry['sha256']}
    return {'all_exact':all(v['matched'] for v in files.values()),
            'files':{str(k):v for k,v in files.items()}}


def require_roles(request):
    pins={local(e['path']).resolve():e['sha256'] for e in request['input_files']}
    for role,path_text in request['roles'].items():
        path=local(path_text)
        if not path.is_absolute() or path.resolve() not in pins:raise ValueError('unpinned role: '+role)
    bundle=local(request['native_bundle'])
    for name in ('manifest.json','native_prepared.xml','prepared_model_arrays.npz','contract.json','walk003/native_original.npz'):
        if (bundle/name).resolve() not in pins:raise ValueError('unpinned native loader input: '+name)
    manifest=read(bundle/'manifest.json')
    for name,digest in manifest['meshes'].items():
        path=(bundle/'meshes'/name).resolve()
        if not path.is_relative_to((bundle/'meshes').resolve()) or pins.get(path)!=digest:
            raise ValueError('unbound geometry mesh')
    return pins


def ready(request_path,clearance_path):
    request_path=local(request_path).resolve();clearance_path=local(clearance_path).resolve()
    request=read(request_path);clearance=read(clearance_path);request_sha=sha(request_path)
    if clearance.get('root_selected_single_run') is not True:raise ValueError('actual run not selected')
    if clearance['request_sha256']!=request_sha:raise ValueError('clearance request changed')
    review_path=local(clearance['review']['path']);review=read(review_path)
    if sha(review_path)!=clearance['review']['sha256'] or review[clearance['review']['pass_field']] is not True:
        raise ValueError('actual final review missing or failed')
    subject=review['request_subject']
    if local(subject['path']).resolve()!=request_path or subject['sha256']!=request_sha:
        raise ValueError('literal reviewed request subject differs')
    expected={'requested_controls':1819,'main_controls':1569,'hold_controls':250,
              'native_step_budget':18190,'serialization_budget':4,'model_inference_calls':0,
              'optimizer_updates':0,'other_oracle_native_steps':0,'epoch_lead_ns':200000000}
    if any(type(request.get(k)) is not int or request[k]!=v for k,v in expected.items()):
        raise ValueError('fixed selected request budget changed')
    pins=require_roles(request)
    source=Path(__file__).resolve().parent
    for path in source.glob('*.py'):
        if path.resolve() not in pins:raise ValueError('unbound imported source')
    checks=pin_check(request['input_files'])
    if not checks['all_exact']:raise ValueError('frozen input changed')
    return request,request_sha,checks


def worker_verdict(cleanup,report,pid):
    return bool(cleanup is not None and cleanup.get('normal_exit') is True and
        cleanup.get('pid')==pid and report is not None and report.get('pid')==pid and
        report.get('failure') is None and report.get('overflow') is None and
        report.get('native_steps')==0 and report.get('model_calls')==0)
