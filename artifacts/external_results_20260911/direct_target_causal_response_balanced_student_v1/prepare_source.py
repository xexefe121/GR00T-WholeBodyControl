"""Freeze source-only preparation/proposal using existing qualified metadata.

No checkpoint loading, corpus generation, task model calls or fit dispatch.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'direct_target_causal_context_study_v2'
MATH=NEW/'direct_target_response_balance_math_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()
def write(p,value):
    with Path(p).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def subject(p,pass_field=None,required=None):
    value=dict(path=Path(p).as_posix(),sha256=sha(p))
    if pass_field is not None:value['pass_field']=pass_field
    if required is not None:value['required_fields']=required
    return value

prior=read(OLD/'training_request.json')
subjects={k:copy.deepcopy(prior['subjects'][k]) for k in ('full_state_root_audit','full_state_data_review',
    'full_state_root_owner','coefficient_source','context_proof')}
subjects.update(checkpoint=subject(OLD/'fit/causal/student_head.pt'),
    normalization=subject(OLD/'fit/shared/normalization.npz'),
    fit_report=subject(OLD/'fit/causal/report.json'),
    fit_owner=subject(OLD/'causal_owner_completion_verification.json','owner_verification_passed',{'condition':'causal','paired_completion_passed':True}),
    fit_audit=subject(NEW/'direct_target_context_pair_fit_independent_v1/results_v1/report.json','evidence_audit_passed',{'export_qualified':True,'paired_complete':True}),
    source_training_request=subject(OLD/'training_request.json'),source_frozen_inputs=subject(OLD/'training_frozen_inputs.json'),
    context_manifest=subject(OLD/'fit/shared/output_manifest.json'),
    context_alignment=subject(OLD/'fit/shared/context_alignment.json'),
    energy_source=subject(MATH/'energy_source.json'),balance_source_preparation=subject(MATH/'source_preparation.json','source_preparation_passed'),
    balance_source_review=subject(NEW/'direct_target_response_balance_root_review_v1/review.json','math_review_pass'),
    semantics_review=subject(NEW/'direct_target_context_saved_semantics_review_v1/results_v1/report.json','passed',
                            {'context_condition':'causal','features':1323,'moving_controls':53}))
if subjects['checkpoint']['sha256']!='10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd':raise ValueError('Source checkpoint identity.')
context_paths={key:(OLD/'fit/shared'/name).as_posix() for key,name in (
    ('normalization','normalization.npz'),('manifest','output_manifest.json'),('nominal_context','nominal_context.npy'),
    ('center_context','center_context.npy'),('physical_context','physical_context.npy'),('data_identities','data_identities.npz'),
    ('context_alignment','context_alignment.json'),('schedule_centers','schedule_centers.npy'),('schedule_axes','schedule_axes.npy'))}
budgets=dict(training_forward_rows=44058000,training_forward_calls=9000,training_updates=3000,
    diagnostic_Torch_rows=1470280,diagnostic_Torch_calls=5748,diagnostic_ORT_rows=367570,diagnostic_ORT_calls=1437,
    calibration_forward_calls=0,calibration_gradient_calls=0,native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
proposal=dict(kind='causal_response_balanced_continuation',root_selected=False,source_preparation_selected=True,
    condition='causal',updates=3000,ordinary_start_step=68000,ordinary_final_step=71000,optimizer_start_step=3000,
    optimizer_final_step=6000,fresh_optimizer=False,coefficient=1.8188207859141674,coefficient_recalibration=False,
    learning_rate=[1e-5,1e-6],weight_decay=1e-5,gradient_clip=10.,paths=prior['paths'],full_state_paths=prior['full_state_paths'],
    subjects=subjects,context_paths=context_paths,
    restoration_predictions={name:(OLD/'fit/causal'/('final_GPU32_'+name+'.npy')).as_posix() for name in ('nominal','full_state','physical')},
    runtime=prior['runtime'],budgets=budgets,features=1323,context_order='previous_action23_then_incoming_history300',
    first_layer_execution='split_contiguous_1000_plus_323',export_first_layer_execution='monolithic_float64_1323',
    initial_parity_tolerance_rad=1e-5,initial_byte_gate_required=False,parity_tolerance_rad=1e-5,
    context_and_normalization_reused=True,no_checkpoint_selection=True,automatic_retry=False,
    objective='N + fixed_coefficient * mean54(original_response_cell_MSE * fixed_teacher_group_weight) + P',
    group_weights=read(MATH/'energy_source.json')['group_weights'],
    selected_protocol_supersedes_math_proposal_learning_rate=True,
    no_causal_advantage_over_extra_epochs_or_warming_claim=True,
    pending_before_actual_fit=['independent exact prepared-source review','actual source/input/request/launcher binding and root concrete clearance'])
write(BASE/'training_request_proposal.json',proposal)

tests=ET.parse(BASE/'synthetic_tests_v3.xml').getroot()
suites=[tests] if tests.tag=='testsuite' else list(tests.iter('testsuite'))
if sum(int(s.get('tests',0)) for s in suites)!=35 or any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped')):
    raise ValueError('Required35 synthetic tests failed.')
prepared=BASE/'source_prepared_v1';prepared.mkdir(exist_ok=False)
source_map={};originals={};math_exact={}
for path in sorted((BASE/'source_draft_v1').glob('*.py')):
    ast.parse(path.read_text(encoding='utf-8'),filename=path.name)
    shutil.copyfile(path,prepared/path.name);source_map[path.name]=sha(prepared/path.name)
    if (OLD/'source_snapshot_v1'/path.name).exists() and sha(OLD/'source_snapshot_v1'/path.name)==source_map[path.name]:originals[path.name]=source_map[path.name]
    if (MATH/'source_prepared_v1'/path.name).exists() and sha(MATH/'source_prepared_v1'/path.name)==source_map[path.name]:math_exact[path.name]=source_map[path.name]
if not {'balance_contract.py','balanced_response_objective.py','direct_objective.py','full_state_contract.py','full_state_objective.py','test_balance_math.py'}.issubset(math_exact):
    raise ValueError('Reviewed math modules changed.')
derivation=dict(source_sha256=source_map,original_modules_byte_exact=originals,reviewed_math_modules_byte_exact=math_exact,
    changed_execution='Warm exact1323 actor/AdamW/RNG restoration; fixed response weights only; all diagnostics retained',
    task_checkpoint_loaded=False,task_model_calls=0,task_gradient_calls=0,optimizer_updates=0,native_steps=0)
write(BASE/'source_derivation.json',derivation)
receipt=dict(source_preparation_passed=True,preparation_only=True,root_selected=False,
    source_directory=prepared.as_posix(),source_sha256=source_map,original_modules_byte_exact=originals,
    reviewed_math_modules_byte_exact=math_exact,synthetic_CPU_tests_passed=35,
    subjects={p.name:subject(p) for p in (BASE/'training_request_proposal.json',BASE/'DESIGN.md',BASE/'OUTPUT_SCHEMA.md',
         BASE/'synthetic_tests_v3.xml',BASE/'source_derivation.json',BASE/'derive_driver.py',Path(__file__),
         MATH/'source_preparation.json',MATH/'energy_source.json')},
    qualified_sources=subjects,task_model_calls=0,task_gradient_calls=0,task_checkpoint_loads=0,
    task_prediction_arrays_loaded=0,actual_data_regenerated=False,optimizer_updates=0,native_steps=0,
    actual_request_created=False,execution_dispatched=False)
write(BASE/'source_preparation.json',receipt)
print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),
    proposal_sha256=sha(BASE/'training_request_proposal.json'),source_files=len(source_map),
    original_modules_byte_exact=len(originals),synthetic_tests=35)))
