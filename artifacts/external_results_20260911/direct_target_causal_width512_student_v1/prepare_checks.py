"""Prepare saved-only schedule proof and metadata-only helper checks; no task model."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil
import sys
import numpy as np

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'direct_target_causal_response_balanced_student_v2'
sys.path.insert(0,str(BASE/'source_draft_v1'))
from full_state_contract import verify_schedule

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as stream:stream.write(v if isinstance(v,str) else json.dumps(v,indent=2,allow_nan=False)+'\n')

def helpers():
    substitutions={
        '9000,44058000':'30000,146860000',
        'ordinary_start_step=68000,ordinary_final_step=71000,additional_updates=3000':'ordinary_start_step=71000,ordinary_final_step=81000,additional_updates=10000',
        'optimizer_start_step=3000,optimizer_step=6000':'optimizer_start_step=6000,optimizer_step=16000',
        "training_first_layer_execution='split_contiguous_1000_plus_323'":"training_first_layer_execution='split_old256_new256_original1000_plus323',hidden_width=512,expansion_seed=20260912,architecture=[1323,512,512,23]"}
    for name in ('verify_completed.py','test_execution_helpers.py'):
        text=(OLD/name).read_text(encoding='utf-8-sig')
        for before,after in substitutions.items():
            assert before in text,(name,before)
            text=text.replace(before,after)
        write(BASE/name,text)
    text=(OLD/'test_launcher.ps1').read_text(encoding='utf-8-sig')
    for before,after in [('causal_response_balanced_continuation','causal_width512_warm_continuation'),
        ('"updates":3000','"updates":10000'),('"ordinary_final_step":71000','"ordinary_final_step":81000')]:
        assert before in text
        text=text.replace(before,after)
    write(BASE/'test_launcher.ps1',text)

def schedule():
    request=read(BASE/'training_request_proposal.json')
    paths={('full_'+k):v for k,v in request['schedule_paths'].items()}
    paths.update({('prior_'+k):request['context_paths'][k] for k in ('schedule_centers','schedule_axes')})
    paths['generation_request']=request['full_state_paths']['request']
    centers_path=Path(request['full_state_paths']['manifest']).parent/'centers.npz'
    paths['centers_metadata']=centers_path.as_posix()
    pins={Path(p).as_posix():sha(p) for p in paths.values()}
    arrays={key:np.load(path,allow_pickle=False) for key,path in paths.items() if key.startswith(('full_','prior_'))}
    with np.load(centers_path,allow_pickle=False) as a:metadata={key:a[key].copy() for key in ('dataset','control','axis_group')}
    for prefix,count in [('full',10000),('prior',3000)]:
        verify_schedule(arrays[prefix+'_schedule_centers'],arrays[prefix+'_schedule_axes'],
            metadata['dataset'],metadata['control'],metadata['axis_group'],updates=count)
    for key in ('schedule_centers','schedule_axes'):
        full=arrays['full_'+key];prior=arrays['prior_'+key]
        assert full[:3000].dtype==prior.dtype and full[:3000].tobytes()==prior.tobytes()
    assert all(sha(path)==digest for path,digest in pins.items())
    write(BASE/'saved_schedule_proof.json',dict(passed=True,full_updates=10000,prior_updates=3000,
        pairs_per_update=864,endpoint_rows_per_update=1728,first3000_centers_byte_exact=True,
        first3000_axes_byte_exact=True,all54_cells_and_groups_valid=True,schedule_generated=False,
        paths=paths,input_sha256=pins,all_inputs_rehashed=True,source_sha256=sha(__file__),
        task_checkpoint_loads=0,task_model_calls=0,optimizer_updates=0,native_steps=0))
    print(json.dumps(dict(saved_schedule_proof=sha(BASE/'saved_schedule_proof.json'))))

def sources():
    files=sorted((BASE/'source_draft_v1').glob('*.py'))
    for p in files:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    write(BASE/'source_static_checks.json',dict(passed=True,source_sha256={p.name:sha(p) for p in files},
        all_python_ASTs_valid=True,files=len(files),task_checkpoint_loads=0,task_model_calls=0,native_steps=0))

if __name__=='__main__':
    command=sys.argv[1:]
    if command==['helpers']:helpers()
    elif command==['schedule']:schedule()
    elif command==['sources']:sources()
    else:raise SystemExit('Use helpers, schedule or sources; never dispatches.')
