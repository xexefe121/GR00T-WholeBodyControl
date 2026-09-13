"""Draft request/launcher generator; no model import or automatic fit launch.

Source-writing preparation may run independently. Input rehash/freeze is deferred
until the clock pause ends and all concrete source qualifications are available.
"""
from pathlib import Path
import hashlib
import json
import shutil

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'direct_target_causal_response_balanced_student_v2'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(value if isinstance(value,str) else json.dumps(value,indent=2,allow_nan=False)+'\n')
def subject(path,field=None):
    value=dict(path=Path(path).as_posix(),sha256=sha(path))
    if field:value['pass_field']=field
    return value


def prepare_launcher():
    text=(OLD/'run_fit_durable_v1.ps1').read_text()
    replacements={
        'causal_response_balanced_continuation':'causal_width512_warm_continuation',
        '$clear.updates -ne 3000':'$clear.updates -ne 10000',
        '$clear.ordinary_final_step -ne 71000':'$clear.ordinary_final_step -ne 81000',
        '$clear.optimizer_start_step -ne 3000':'$clear.optimizer_start_step -ne 6000',
        '$clear.optimizer_final_step -ne 6000':'$clear.optimizer_final_step -ne 16000',
        '$request.updates -ne 3000':'$request.updates -ne 10000',
        '$request.ordinary_final_step -ne 71000':'$request.ordinary_final_step -ne 81000',
        'updates=3000;ordinary_start_step=68000;ordinary_final_step=71000':'updates=10000;ordinary_start_step=71000;ordinary_final_step=81000',
        "optimizer_start_step=3000;optimizer_final_step=6000":"optimizer_start_step=6000;optimizer_final_step=16000",
        '$report.ordinary_start_step -ne 68000':'$report.ordinary_start_step -ne 71000',
        '$report.ordinary_final_step -ne 71000':'$report.ordinary_final_step -ne 81000',
        '$report.additional_updates -ne 3000':'$report.additional_updates -ne 10000',
        '$report.optimizer_start_step -ne 3000':'$report.optimizer_start_step -ne 6000',
        '$report.optimizer_step -ne 6000':'$report.optimizer_step -ne 16000',
        '$report.counts.training.rows_verified -ne 44058000':'$report.counts.training.rows_verified -ne 146860000',
        '$report.counts.training.calls_verified -ne 9000':'$report.counts.training.calls_verified -ne 30000'}
    for old,new in replacements.items():
        if old not in text:raise ValueError('Launcher source fragment missing: '+old)
        text=text.replace(old,new)
    marker="    if($report.fixed_full_state_coefficient"
    assert text.count(marker)==1
    text=text.replace(marker,"    if($report.hidden_width -ne 512 -or $report.expansion_seed -ne 20260912 -or $report.training_first_layer_execution -ne 'split_old256_new256_original1000_plus323'){throw 'Width512 geometry differs.'}\n"+marker)
    write(BASE/'run_fit_durable_v1.ps1',text)
    with (BASE/'read_fit_progress_v1.ps1').open('xb') as stream:stream.write((OLD/'read_fit_progress_v1.ps1').read_bytes())


def prepare_request():
    """Execute only after the clock pause: hashes qualified metadata/output files."""
    request=read(OLD/'training_request.json')
    request.update(kind='causal_width512_warm_continuation',root_selected=False,
        updates=10000,ordinary_start_step=71000,ordinary_final_step=81000,
        optimizer_start_step=6000,optimizer_final_step=16000,fresh_optimizer=False,
        architecture=[1323,512,512,23],old_hidden_width=256,new_hidden_width=512,expansion_seed=20260912,
        first_layer_execution='split_old256_new256_original1000_plus323',
        old_optimizer_blocks_exact=True,new_optimizer_moments_zero=True,shared_optimizer_step_for_new_entries=6000,
        learning_rate=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],cosine_updates=9750,cosine_inclusive=[1e-5,1e-6]),
        budgets=dict(training_forward_rows=146860000,training_forward_calls=30000,training_updates=10000,
            diagnostic_Torch_rows=1470280,diagnostic_Torch_calls=5748,diagnostic_ORT_rows=367570,diagnostic_ORT_calls=1437,
            calibration_forward_calls=0,calibration_gradient_calls=0,native_calls=0,BFM_calls=0,manual_export_trace_calls=0),
        pending_before_actual_fit=['clock process absence','completed width source/initialization/export tests and independent review',
                                   'actual root selection and concrete request/source/runtime/launcher clearance'])
    # Preserve all unchanged lineage; replace consumed start subjects explicitly.
    subjects=request['subjects']
    for name,path,field in [
        ('checkpoint',OLD/'fit/student_head.pt',None),('fit_report',OLD/'fit/report.json',None),
        ('fit_owner',OLD/'owner_completion_verification.json','owner_verification_passed'),
        ('fit_audit',NEW/'direct_target_response_balanced_fit_independent_v3/results_v1/report.json','evidence_audit_passed'),
        ('normalization',OLD/'fit/shared/normalization.npz',None),
        ('context_alignment',OLD/'fit/shared/context_alignment.json',None),
        ('context_manifest',OLD/'fit/shared/output_manifest.json',None),
        ('warm_source_frozen_inputs',OLD/'training_frozen_inputs.json',None),
        ('width_initializer_preparation',NEW/'direct_target_causal_width512_preparation_v1/source_preparation.json','source_preparation_passed')]:
        subjects[name]=subject(path,field)
    request['context_paths']={key:(OLD/'fit/shared'/Path(path).name).as_posix() for key,path in request['context_paths'].items()}
    request['context_paths']['manifest']=(OLD/'fit/shared/output_manifest.json').as_posix()
    request['restoration_predictions']={corpus:(OLD/'fit'/('final_GPU32_'+corpus+'.npy')).as_posix() for corpus in ('nominal','full_state','physical')}
    request['schedule_paths']={name:(NEW/'direct_target_full_state_student_v1/fit'/(name+'.npy')).as_posix()
                               for name in ('schedule_centers','schedule_axes')}
    # Stale preparation/review roles remain named as inherited provenance only;
    # they cannot qualify new source or authorize execution.
    for key in ('source_preparation','source_review','root_data_export_review','repair_review'):
        if key in subjects:subjects['inherited_'+key]=subjects.pop(key)
    request.pop('source_preparation_sha256',None)
    write(BASE/'training_request_proposal.json',request)
    print(json.dumps(dict(request_proposal_sha256=sha(BASE/'training_request_proposal.json'),actual_fit_selected=False)))


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['launcher']:prepare_launcher()
    elif sys.argv[1:]==['request']:prepare_request()
    else:raise SystemExit('Use launcher or request; no dispatch.')
