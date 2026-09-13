"""Completed witness plus one selected canonical launch. Saved arrays/source only."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=NEW/'direct_target_student_evaluation_v1';OUT=Path(__file__).resolve().parent
pins={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def bind(p,expected=None):
    p=Path(p).resolve();key=p.as_posix()
    if key in pins:
        if expected is not None:assert pins[key]==expected,(p,pins[key],expected)
        return p
    s=sha(p)
    if expected is not None:assert s==expected,(p,s,expected)
    pins[key]=s;return p
def read(p,expected=None):return json.loads(bind(p,expected).read_text(encoding='utf-8-sig'))
def subject(p):p=bind(p);return dict(path=p.as_posix(),sha256=pins[p.as_posix()])
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
p=argparse.ArgumentParser();p.add_argument('--binding-sha',required=True);p.add_argument('--receipt-sha',required=True);p.add_argument('--pins',type=int,required=True);a=p.parse_args()
assert not (OUT/'review.json').exists()
binding=read(BASE/'evaluation_binding.json',a.binding_sha)
folder=BASE/'evaluation_process';receipt=read(folder/'launch_receipt.json',a.receipt_sha)
assert receipt['kind']=='one_selected_direct_target_evaluation' and receipt['binding_sha256']==a.binding_sha
assert receipt['requested_main_controls']==1569 and receipt['conditional_hold_controls']==250 and receipt['expected_separate_head_calls']==0
assert receipt['hardware_authorized'] is False and receipt['automatic_retry'] is False
assert binding['ordinary_final_step']==5000 and binding['architecture']==[1000,256,256,23]
assert binding['hidden_activation']=='ELU' and binding['head_output']=='normalized_target'
assert binding['controller']=='direct_absolute_target_1000_applied_inverse_learned_prior'
assert binding['span_contract']=='existing_float32_joint_span_promoted_float64'
assert binding['root_authorized_evaluation'] is True and binding['physics_authorized'] is True
assert binding['requested_main_controls']==1569 and binding['conditional_hold_controls']==250 and binding['learned_BFM_calls']==0
for k in ('hardware_authorized','clock_foundation_connected','filters_enabled','compiled_preview_enabled'):assert binding[k] is False,k
assert len(receipt['input_hashes'])==a.pins
for item in binding['input_files']:assert receipt['input_hashes'][item['path']]==item['sha256']
for path,digest in receipt['input_hashes'].items():bind(path,digest)
fit_review=read(NEW/'direct_target_final_fit_review_v1/review.json','5ef6fcd9eeb9371d05b1e345c29a9fd67fcf9bc1832e5893b1b6b2093c5ec1a6')
helper=read(NEW/'direct_target_launch_helpers_review_v1/review.json','574607b86a9ae1414b26afdfd06855fa2d26d1a79b6ac66f5e8049661744e118')
assert helper['source_review_pass'] is True
for name in ('dataset','fit','export'):assert fit_review[name+'_review_pass'] is True
for name in ('head','checkpoint','fit_report','training_request','training_manifest','centers','query250_labels','contract'):
    item=binding[name];bind(item['path'],item['sha256'])
for name in ('head','checkpoint','fit_report','training_request'):assert binding[name]['sha256']==fit_review['subjects'][name]['sha256']
assert binding['training_manifest']['sha256']==fit_review['subjects']['frozen_inputs']['sha256']
assert binding['training_dataset_sha256']==fit_review['training_dataset_sha256']
for role,item in binding['reviews'].items():
    value=read(item['path'],item['sha256'])
    for key in item['pass_field'].split('.'):value=value[key]
    assert value is True,role
assert binding['root_audit']['sha256']==fit_review['subjects']['root_audit']['sha256']
assert read(binding['root_audit']['path'],binding['root_audit']['sha256'])['passed'] is True
source=read(BASE/'source_preparation.json','7391598abf062789b8daae00d85076470b947c0b3036242eccd8b2f5b36bab5f')
for name,digest in source['source_sha256'].items():bind(BASE/'source_draft_v1'/name,digest)
# Check completed, separately selected one-call WSL witness and immutable launch lineage.
wowner=read(BASE/'witness_completion_verification.json','856613585dc82885964eb7e4ff18e5d94d2cd1c3ff351fe7276661e4e9aef2d4')
assert wowner['owner_completion_accounting_passed'] is True and wowner['diagnostic_passed'] is True
assert wowner['raw_python_exit_code']==wowner['diagnostic_exit_code']==0 and wowner['pins_exact']==782
for path,digest in wowner['output_hashes'].items():bind(path,digest)
wfolder=BASE/'witness_process';wreceipt=read(wfolder/'launch_receipt.json','95bfb56d091a060f0f3231471a82d4b5bc399ea8b40e685cff79b8889807bfa1')
wpost=read(wfolder/'postrun_hashes.json');assert wpost==wreceipt['input_hashes']
for path,digest in wpost.items():bind(path,digest)
wx=read(wfolder/'exit.json');wr=read(wfolder/'raw_exit.json');wv=read(wfolder/'diagnostic_verdict.json')
assert wx['exit_code']==wx['raw_child_exit_code']==wr['raw_python_exit_code']==wv['diagnostic_exit_code']==0
assert wx['error'] is None and wx['all_postrun_hashes_exact'] is True and wv['passed'] is True
start=read(wfolder/'start.json');child=read(wfolder/'child.json');absence=read(wfolder/'process_absence.json')
assert absence['wrapper_absent'] is True and absence['child_absent'] is True
assert absence['wrapper_pid']==start['wrapper_pid'] and absence['child_pid']==child['child_pid']
wc=read(wfolder/'launch_clearance.json')
assert wc['launch_receipt_sha256']==start['receipt_sha256']==sha(wfolder/'launch_receipt.json')
assert start['clearance_sha256']==sha(wfolder/'launch_clearance.json')
assert start['review_sha256']==wc['review']['sha256']=='e1f0f57ec82418a6de53e3e0dc3c820ca2dc7febe3cae46c5e0b5708d55b28d5'
read(wc['review']['path'],wc['review']['sha256'])
wp=bind(binding['first_export_witness']['path'],binding['first_export_witness']['sha256'])
assert sha(wp)=='6c4ec2e7d16afa7513eacbaeeb08c80b6bbba8c40eb164c642b6cd87d7e5b195'
w=read(binding['first_export_receipt']['path'],binding['first_export_receipt']['sha256'])
assert w['pass_all'] is True and w['expected_head_calls']==w['attempted_head_calls']==w['returned_head_calls']==1
assert w['BFM_inference_calls']==w['physics_steps']==0 and w['fitting_launched'] is False
assert w['head_sha256']==binding['head']['sha256'] and w['witness_sha256']==sha(wp)
assert w['binding_sha256']=='be14ed9432ae02cbd85d8e301b1a4e76581230c36fa3a5cb441c800fb6c2f543'
assert w['source_sha256']==source['source_sha256']['head_activation_witness.py']
assert w['centers_row']==2038 and w['control']==250 and w['source_frame']==261
assert w['runtime_binary_sha256'] in receipt['input_hashes'].values()
assert w['onnxruntime_version']=='1.23.2' and w['execution_provider']=='CPUExecutionProvider' and w['execution_mode']=='ORT_SEQUENTIAL'
assert w['intra_op_threads']==w['inter_op_threads']==1 and w['pinned_WSL_runtime'] is True
with np.load(wp,allow_pickle=False) as z, np.load(binding['centers']['path'],allow_pickle=False) as c:
    for name,value in [('features',c['features'][2038][np.r_[0:52,75:1023]]),('span',c['joint_span']),('limits',c['joint_limits'])]:assert exact(z[name],value),name
    default=np.asarray(read(binding['contract']['path'])['default_q'],np.float64)
    assert exact(z['default'],default)
    assert z['normalized_target'].dtype==np.float32 and z['normalized_target'].shape==(23,) and np.isfinite(z['normalized_target']).all()
    raw=default+c['joint_span'].astype(np.float64)*z['normalized_target'].astype(np.float64)
    assert exact(z['raw_proposal'],raw) and exact(z['target'],np.clip(raw,c['joint_limits'][:,0],c['joint_limits'][:,1]))
    assert str(z['head_sha256'].item())==binding['head']['sha256'] and str(z['runtime_binary_sha256'].item())==w['runtime_binary_sha256']
spec=importlib.util.spec_from_file_location('reviewed_launch',BASE/'prepare_bound_launcher.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert receipt['exact_wsl_arguments']==m.wsl_arguments(BASE,'evaluation')
assert receipt['exact_wsl_arguments'][2:4]==['--cd','/']
assert (folder/'run.ps1').read_text()==m.run_text(BASE,folder,'evaluation',a.binding_sha)
assert (folder/'run_durable.ps1').read_text()==m.durable_text(folder)
for path in [BASE/'nominal',BASE/'post_lifecycle_hold_5s',BASE/'pilot_outcome.json',*[folder/n for n in ('launch_clearance.json','started.lock','start.json','child.json','exit.json','raw_exit.json','stdout.log','stderr.log')]]:assert not path.exists(),path
bind(__file__)
review=dict(kind='one_selected_direct_target_canonical_prelaunch_review',verdict='CLEAR',passed=True,prelaunch_review_pass=True,
    subjects={name:subject(path) for name,path in [('binding',BASE/'evaluation_binding.json'),('launch_receipt',folder/'launch_receipt.json'),('fit_export_review',NEW/'direct_target_final_fit_review_v1/review.json'),('helper_source_review',NEW/'direct_target_launch_helpers_review_v1/review.json'),('head',binding['head']['path']),('fit_report',binding['fit_report']['path']),('witness',wp),('witness_report',binding['first_export_receipt']['path']),('witness_completion',BASE/'witness_completion_verification.json')]},
    verified_canonical_launch_pins=a.pins,verified_witness_launch_pins=782,source_sha256=source['source_sha256'],input_sha256=pins,
    requested_main_controls=1569,conditional_hold_controls=250,additional_witness_calls=0,learned_BFM_calls=0,
    ordinary_final_step=5000,selected_single_canonical=True,strict_original_native_gates_unchanged=True,
    full_BFM250_prefix_and_actual_query250_head_activation_gates_required=True,
    history='Plant measured history updated once per accepted command; learned phase prior from actual native-clipped target; original startup/terminal BFM raw prior.',
    exact_wsl_arguments=receipt['exact_wsl_arguments'],reviewer_task_model_calls=0,reviewer_ORT_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0,
    independent_actual_physics_and_intent_audits_required=True,hardware_authorized=False,behavioral_qualification=False)
with (OUT/'review.json').open('x') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps({'verdict':'CLEAR','review_sha256':sha(OUT/'review.json'),'launch_pins':a.pins,'all_review_pins':len(pins)}))
