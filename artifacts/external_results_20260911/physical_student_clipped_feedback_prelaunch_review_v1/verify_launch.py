"""Pure final subject/source/launch verification; never executes controller."""
import hashlib
import json
from pathlib import Path
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OWNER=BASE/'physical_student_clipped_feedback_v1';OUT=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def keyfield(v,key):
    for k in key.split('.'):v=v[k]
    return v
pins={}
def add(p,d=None):
    p=Path(p);actual=sha(p)
    if d is not None:assert actual==d,str(p)
    pins[p.as_posix()]=actual;return actual
bp=OWNER/'evaluation_binding.json';lp=OWNER/'evaluation_process/launch_receipt.json';fp=OWNER/'frozen_inputs_v2.json'
add(bp,'f901d23b8ece3471f4f99c1c3b3529d4d5e20d68f4f4490551dbcb3ef8ebc840')
add(lp,'93753131d24a96890c4a78e31e12c4c31f05b248240e08cd9362cc2a4f8a2deb')
add(fp,'04f0f61245f54918a1efcfd4bb871d43322ebde3df8c3a4f1a3a20164f623495')
b=read(bp);l=read(lp);f=read(fp)
assert len(l['input_hashes'])==1619
for p,d in l['input_hashes'].items():add(p,d)
for e in b['input_files']:assert pins[e['path']]==e['sha256']
for p,d in f['input_sha256'].items():assert pins[p]==d
for n,d in f['source_sha256'].items():assert pins[(OWNER/'source_snapshot_v1'/n).as_posix()]==d
assert l['binding_sha256']==sha(bp)
assert b['root_authorized_canonical_evaluation'] is True and b['ordinary_final_step']==75000
assert b['requested_main_controls']==l['requested_main_controls']==1569
assert b['conditional_hold_controls']==l['conditional_hold_controls']==250
assert b['controller']=='learned_native_clipped_component_feedback'
for k in ('unclipped_float32_feedback_preserved','initial_and_terminal_BFM_feedback_unchanged','reused_witness_only'):assert b[k] is True
assert b['additional_witness_calls']==l['expected_separate_head_calls']==0
for k in ('hardware_authorized','filters_enabled','compiled_preview_enabled'):assert b[k] is False
for k,value in [('original_prefix_native_steps_required',2650),('first_changed_previous_action_control',265),('first_changed_actor_lag_action_control',266)]:assert b[k]==value
known=dict(head='fb856003734acc0338586482968a7e31553a11826e549a8b486e0662e4934e81',checkpoint='9f31d74855c28a57231c51e0a652a5f2eeeaac87a3c5aff032d59ba5ca981e34',fit_report='5153ef3e193059062cf071b2b82c1961f7465683e8d8c9e5471dd89a27b25a1b',first_export_witness='fa2dad8a24f43f577ec1120a86e41d7e76f6d09805cc0e637b14907c0a56e088',first_export_receipt='9bc2c1f112ae321b9ef5a4a6661afe3c556381b34ee7857ad9cca52fce58e96d',original_raw_trace='38428ae27055be7c2e771c5a23056c858f004ef38f260a0b45a57662d8d0aae3')
for k,d in known.items():assert b[k]['sha256']==d and pins[b[k]['path']]==d
for role,e in b['reviews'].items():
    add(e['path'],e['sha256']);assert keyfield(read(e['path']),e['pass_field']) is True,role
assert b['reviews']['source']['sha256']=='8c220437e60f84e492f11580fe16f11a04da2ebf72da07ac019b175ab5119c11'
w=read(b['first_export_receipt']['path'])
assert w['pass_all'] is True and w['expected_head_calls']==w['attempted_head_calls']==1
assert w['BFM_inference_calls']==w['physics_steps']==w['fit_head_calls_repeated']==0
bootstrap='Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
assert pins[bootstrap]=='392de6eccb281c41219566c4b7c9c1813b086913f66d861529d4904959168ebc'
linux='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/physical_student_clipped_feedback_v1'
expected=['-d','Ubuntu-22.04','--cd','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof','--','bash','/mnt/z'+bootstrap[2:],'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH='+linux+'/source_snapshot_v1','/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',linux+'/source_snapshot_v1/evaluate_physical_response_student.py']
assert l['exact_wsl_arguments']==expected
run=(OWNER/'evaluation_process/run.ps1').read_text();durable=(OWNER/'evaluation_process/run_durable.ps1').read_text()
for text in ('1569','250','15690','2500','CreateNew','original_prefix265_and_first_feedback_parity.json','first_feedback_actor_input265_parity.json','first_feedback_lag_history266_parity.json','first_feedback_actor_lag266_parity.json'):assert text in run
assert durable.index('$nativeHandle = $taskChild.Handle')<durable.index('$taskChild.WaitForExit()')<durable.index('$exitCode = $taskChild.ExitCode')
assert '-WindowStyle Hidden' in durable and 'postrun_hashes.json' in durable and 'Child exit status unavailable.' in durable
for name in ('nominal','post_lifecycle_hold_5s','pilot_outcome.json','head_witness','evaluation_process/started.lock','evaluation_process/execution.lock','evaluation_process/start.json','evaluation_process/child.json','evaluation_process/stdout.log','evaluation_process/stderr.log','evaluation_process/exit.json'):assert not (OWNER/name).exists(),name
add(Path(__file__))
report=dict(kind='one_selected_clipped_feedback_final_canonical_prelaunch_review',verdict='CLEAR',prelaunch_review_pass=True,canonical_launch_review_pass=True,source_review_pass=True,
    subjects=dict(binding=dict(path=bp.as_posix(),sha256=sha(bp)),launch_receipt=dict(path=lp.as_posix(),sha256=sha(lp)),frozen_inputs=dict(path=fp.as_posix(),sha256=sha(fp)),**{k:b[k] for k in known}),
    source_sha256=f['source_sha256'],input_sha256=pins,launch_pins_verified=1619,
    exact_wsl_arguments=expected,requested_main_controls=1569,conditional_hold_controls=250,
    additional_witness_calls=0,head_and_witness_reused=True,root_selected_one_run=True,
    original_prefix_controls=265,original_prefix_native_steps=2650,first_changed_prior_control=265,first_changed_actor_lag_control=266,
    original_physics_constraints_and_initialization_unchanged=True,output_and_execution_locks_absent=True,
    preservation='Raw proposals/actions retained separately from component-selected feedback; partial native and returned-output evidence, exact transition failures and incomplete exits are preserved.',
    limitation='Controlled later feedback experiment only. Early pre-clipping drift remains unexplained; no native/full-body/hardware qualification is inferred.',
    reviewer_model_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0,hardware_authorized=False)
with (OUT/'review.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(dict(verdict='CLEAR',review_sha256=sha(OUT/'review.json'),pins=len(pins))))
