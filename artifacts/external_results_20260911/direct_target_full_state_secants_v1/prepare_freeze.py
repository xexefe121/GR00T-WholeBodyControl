"""Freeze actual selected inputs and source; no probe/model/native calls."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def windows(value):
    if value.startswith('/mnt/') and value[6:7]=='/':return value[5].upper()+':'+value[6:]
    return value
if (BASE/'request.json').exists():raise RuntimeError('Preserve previous freeze')
pre=read(BASE/'input_preflight.json');config=read(BASE/'input_config.json')
assert pre['passed'] and pre['task_feature_calls']==pre['task_map_calls']==pre['task_probe_rows']==0
assert pre['config_sha256']==sha(BASE/'input_config.json')
sources={str(p):sha(p) for p in sorted((BASE/'source_snapshot_v1').glob('*.py'))}
inputs={windows(p):h for p,h in pre['input_sha256'].items()}
for path,digest in inputs.items():assert sha(Path(path))==digest
tests={'passed':True,'tests_passed':14,'failures':0,'skips':0,'source_sha256':sources,
    'execution':'Pinned WSL unittest discover; observed exit0;14 synthetic tests passed0.153s',
    'task_probe_rows':0,'model_calls':0,'native_steps':0,'ps_ast_parse_passed':['run_generation_durable.ps1','read_progress.ps1']}
write(BASE/'tests_final.json',tests)
bootstrap=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh')
assert sha(bootstrap)=='392de6eccb281c41219566c4b7c9c1813b086913f66d861529d4904959168ebc'
launch={**inputs,**sources}
for path in (bootstrap,BASE/'run_generation_durable.ps1',BASE/'read_progress.ps1',BASE/'ROOT_SELECTION.md',BASE/'DESIGN.md',
    BASE/'tests_final.json',BASE/'input_config.json',BASE/'input_preflight.json',BASE/'prepare_freeze.py',BASE/'source_preparation.json'):
    launch[str(path)]=sha(path)
runtime=pre['runtime_sha256']
python_candidates=[p for p in runtime if p.endswith('/bin/python'+'.'.join(pre['versions']['python'].split('.')[:2]))]
assert len(python_candidates)==1
numpy_candidates=[p for p in runtime if p.endswith('/site-packages/numpy/__init__.py')]
scipy_candidates=[p for p in runtime if p.endswith('/site-packages/scipy/__init__.py')]
assert len(numpy_candidates)==len(scipy_candidates)==1
request={'kind':'selected_one_full58_saved_secant_generation','generation_selected':True,'rows':3057,'axes':58,'signed_rows':354612,
    'old_overlap_rows':140622,'added_rows':213990,'pure_feature_map_calls':357669,'model_calls':0,'BFM_calls':0,'native_steps':0,'optimizer_updates':0,
    'paths':config['paths'],'input_sha256':inputs,'source_sha256':sources,'runtime_sha256':runtime,'windows_launch_sha256':launch,
    'archive_schemas':pre['archive_schemas'],'scipy_version':pre['versions']['scipy'],
    'runtime_identity':{'python_version':pre['versions']['python'],'python_executable':python_candidates[0],
        'numpy_module':numpy_candidates[0],'scipy_module':scipy_candidates[0]},
    'preflight_sha256':sha(BASE/'input_preflight.json'),'selection_sha256':sha(BASE/'ROOT_SELECTION.md'),
    'source_draft_preparation_sha256':sha(BASE/'source_preparation.json'),'tests_sha256':sha(BASE/'tests_final.json'),
    'radii_contract':'Fixed .001m/.01rad/.01rad/.05mps/.25radps/.01nativecap; no shrink/drop/state clamp',
    'output':str(BASE/'generation')}
write(BASE/'request.json',request)
(BASE/'process').mkdir(exist_ok=False)
receipt={'request_sha256':sha(BASE/'request.json'),'launcher_sha256':sha(BASE/'run_generation_durable.ps1'),
    'windows_launch_pin_count':len(launch),'runtime_pin_count':len(runtime),'source_count':len(sources),
    'source_sha256':sources,'schema_preflight_passed':True,'final_tests_passed':True,'generation_executed':False,
    'required_final_review_fields':['passed=true','request_sha256','launcher_sha256']}
write(BASE/'launch_receipt.json',receipt)
print(json.dumps(receipt,indent=2))
