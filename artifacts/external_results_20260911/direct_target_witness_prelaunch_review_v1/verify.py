"""Concrete one-call witness review. Hash/source/JSON checks only."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=NEW/'direct_target_student_evaluation_v1';OUT=Path(__file__).resolve().parent
pins={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def bind(p,expected=None):
    p=Path(p).resolve();s=sha(p)
    if expected is not None:assert s==expected,(p,s,expected)
    if p.as_posix() in pins:assert pins[p.as_posix()]==s,p
    pins[p.as_posix()]=s;return p
def read(p,expected=None):return json.loads(bind(p,expected).read_text(encoding='utf-8-sig'))
def subject(p):p=bind(p);return dict(path=p.as_posix(),sha256=pins[p.as_posix()])
def has(v,d):return any(has(x,d) for x in v.values()) if isinstance(v,dict) else any(has(x,d) for x in v) if isinstance(v,list) else v==d
assert not (OUT/'review.json').exists()
binding=read(BASE/'witness_binding.json','be14ed9432ae02cbd85d8e301b1a4e76581230c36fa3a5cb441c800fb6c2f543')
receipt=read(BASE/'witness_process/launch_receipt.json','95bfb56d091a060f0f3231471a82d4b5bc399ea8b40e685cff79b8889807bfa1')
assert len(binding['input_files'])==776 and len(receipt['input_hashes'])==782
assert receipt['binding_sha256']==sha(BASE/'witness_binding.json')
assert receipt['kind']=='one_selected_direct_target_witness'
assert receipt['expected_separate_head_calls']==1 and receipt['requested_main_controls']==receipt['conditional_hold_controls']==0
assert receipt['hardware_authorized'] is False and receipt['automatic_retry'] is False
assert binding['ordinary_final_step']==5000 and binding['features'] if 'features' in binding else True
assert binding['root_authorized_witness'] is True and binding['physics_authorized'] is False
assert binding['expected_head_calls']==1 and binding['learned_BFM_calls']==0
assert binding['architecture']==[1000,256,256,23] and binding['hidden_activation']=='ELU' and binding['head_output']=='normalized_target'
assert binding['span_contract']=='existing_float32_joint_span_promoted_float64'
for name in ('hardware_authorized','clock_foundation_connected','filters_enabled','compiled_preview_enabled'):assert binding[name] is False,name
fit_review=read(NEW/'direct_target_final_fit_review_v1/review.json','5ef6fcd9eeb9371d05b1e345c29a9fd67fcf9bc1832e5893b1b6b2093c5ec1a6')
helper=read(NEW/'direct_target_launch_helpers_review_v1/review.json','574607b86a9ae1414b26afdfd06855fa2d26d1a79b6ac66f5e8049661744e118')
assert helper['source_review_pass'] is True
for name in ('dataset','fit','export'):assert fit_review[name+'_review_pass'] is True
for entry in binding['input_files']:
    assert receipt['input_hashes'][entry['path']]==entry['sha256']
for path,digest in receipt['input_hashes'].items():bind(path,digest)
for name in ('head','checkpoint','fit_report','training_request','training_manifest','centers','query250_labels','contract'):
    entry=binding[name];bind(entry['path'],entry['sha256'])
assert binding['head']['sha256']==fit_review['subjects']['head']['sha256']
assert binding['checkpoint']['sha256']==fit_review['subjects']['checkpoint']['sha256']
assert binding['fit_report']['sha256']==fit_review['subjects']['fit_report']['sha256']
assert binding['training_request']['sha256']==fit_review['subjects']['training_request']['sha256']
assert binding['training_manifest']['sha256']==fit_review['subjects']['frozen_inputs']['sha256']
assert binding['training_dataset_sha256']==fit_review['training_dataset_sha256']
for role,entry in binding['reviews'].items():
    r=read(entry['path'],entry['sha256']);value=r
    for key in entry['pass_field'].split('.'):value=value[key]
    assert value is True,role
root=read(binding['root_audit']['path'],binding['root_audit']['sha256']);assert root['passed'] is True
assert binding['root_audit']['sha256']==fit_review['subjects']['root_audit']['sha256']
for name in ('fit_report','head','checkpoint'):assert has(root,binding[name]['sha256'])
source=read(BASE/'source_preparation.json','7391598abf062789b8daae00d85076470b947c0b3036242eccd8b2f5b36bab5f')
for name,digest in source['source_sha256'].items():bind(BASE/'source_draft_v1'/name,digest)
# Pure launcher-text constructors, no model or subprocess calls.
spec=importlib.util.spec_from_file_location('reviewed_launch',BASE/'prepare_bound_launcher.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
folder=BASE/'witness_process'
assert receipt['exact_wsl_arguments']==m.wsl_arguments(BASE,'witness')
assert receipt['exact_wsl_arguments'][2:4]==['--cd','/']
assert (folder/'run.ps1').read_text()==m.run_text(BASE,folder,'witness',sha(BASE/'witness_binding.json'))
assert (folder/'run_durable.ps1').read_text()==m.durable_text(folder)
bind('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh','392de6eccb281c41219566c4b7c9c1813b086913f66d861529d4904959168ebc')
for p in [BASE/'head_witness',*[folder/n for n in ('launch_clearance.json','started.lock','start.json','child.json','exit.json','raw_exit.json','stdout.log','stderr.log')]]:assert not p.exists(),p
bind(__file__)
review=dict(kind='one_selected_direct_target_WSL_witness_prelaunch_review',verdict='CLEAR',passed=True,prelaunch_review_pass=True,
    subjects={name:subject(path) for name,path in [('binding',BASE/'witness_binding.json'),('launch_receipt',folder/'launch_receipt.json'),('fit_export_review',NEW/'direct_target_final_fit_review_v1/review.json'),('helper_source_review',NEW/'direct_target_launch_helpers_review_v1/review.json'),('head',binding['head']['path']),('fit_report',binding['fit_report']['path'])]},
    verified_launch_pins=782,source_sha256=source['source_sha256'],input_sha256=pins,
    expected_head_calls=1,expected_BFM_calls=0,expected_native_steps=0,ordinary_final_step=5000,
    exact_wsl_arguments=receipt['exact_wsl_arguments'],selected_single_witness=True,
    future_canonical_requires_successful_witness_and_concrete_prelaunch_review=True,
    reviewer_task_model_calls=0,reviewer_ORT_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0,
    hardware_authorized=False,behavioral_qualification=False)
with (OUT/'review.json').open('x') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps({'verdict':'CLEAR','review_sha256':sha(OUT/'review.json'),'launch_pins':782}))
