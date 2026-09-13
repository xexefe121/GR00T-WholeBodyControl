"""Pin an unselected request proposal; never creates execution clearance or a model."""
import json
import hashlib
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
SOURCE=BASE/'source_draft_v1'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def item(path):return dict(path=Path(path).as_posix(),sha256=sha(path))

def main():
    original=ROOT/'direct_target_full_state_student_v1'
    old=read(original/'training_request.json');receipt=read(original/'training_frozen_inputs.json')
    subjects=dict(old['subjects'])
    subjects.update(checkpoint=item(original/'fit/student_head.pt'),normalization=item(original/'fit/normalization.npz'),
        fit_report=item(original/'fit/report.json'),coefficient_source=item(original/'fit/coefficient.json'),
        fit_owner=item(original/'owner_completion_verification.json'),
        fit_audit=item(ROOT/'direct_target_full_state_fit_independent_v1/results_v1/report.json'))
    subjects['fit_audit'].update(pass_field='evidence_audit_passed',required_fields=dict(export_qualified=True))
    subjects['fit_owner'].update(pass_field='owner_verification_passed')
    paths={key:(original/'fit'/('final_GPU32_'+key+'.npy')).as_posix() for key in ('nominal','full_state','physical')}
    schedule={key:(original/'fit'/('schedule_'+key+'.npy')).as_posix() for key in ('centers','axes')}
    budgets=dict(training_forward_rows=88116000,training_forward_calls=18000,training_updates=6000,
        diagnostic_Torch_rows=2940560,diagnostic_Torch_calls=11496,diagnostic_ORT_rows=735140,
        diagnostic_ORT_calls=2874,calibration_forward_calls=0,calibration_gradient_calls=0,
        native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
    proposal=dict(kind='matched_causal_context_utility_study',root_selected=False,source_preparation_selected=True,
        conditions=['blinded','causal'],updates_per_condition=3000,ordinary_final_step=68000,
        coefficient=1.8188207859141674,coefficient_recalibration=False,learning_rate=[1e-5,1e-6],
        paths=old['paths'],full_state_paths=old['full_state_paths'],subjects=subjects,
        restoration_predictions=paths,schedule_paths=schedule,runtime=old['runtime'],budgets=budgets,
        context_order='previous_action23_then_incoming_history300',context_std_floor=.05,
        original_features=1000,context_features=323,features=1323,
        no_checkpoint_selection=True,no_controller_selected=True,actual_execution_clearance_absent=True,
        pending_before_actual_fit=['saved context alignment proof and independent review','concrete source/input/request freeze','root fit selection and final launcher review'])
    pins=dict(receipt['input_sha256'])
    for subject in subjects.values():pins[Path(subject['path']).as_posix()]=subject['sha256']
    for path in list(paths.values())+list(schedule.values()):pins[path]=sha(path)
    pins[(original/'training_request.json').as_posix()]=sha(original/'training_request.json')
    pins[(original/'training_frozen_inputs.json').as_posix()]=sha(original/'training_frozen_inputs.json')
    write_new(BASE/'training_request_proposal.json',proposal)
    write_new(BASE/'preparation_inputs.json',dict(input_sha256=pins,
        training_request_proposal_sha256=sha(BASE/'training_request_proposal.json'),
        inherited_frozen_receipt_sha256=sha(original/'training_frozen_inputs.json'),
        all_inputs_rehashed_now=False,actual_execution_authorized=False))
    print(json.dumps(dict(proposal_sha256=sha(BASE/'training_request_proposal.json'),input_pins=len(pins),
        task_model_calls=0,native_steps=0,training_selected=False)))

if __name__=='__main__':main()
