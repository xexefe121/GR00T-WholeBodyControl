"""Synthetic request and lineage rejection tests, without model/task data reads."""
import copy
import pytest
from balance_contract import GROUP_WEIGHTS
from recovery_release import recovery_request_identity, selected_rates, SOURCE_CHECKPOINT

def request():
    return dict(kind='qualified_width251_recovery_warm512_fit', root_selected=True, condition='causal',
        updates=10000, ordinary_start_step=81000, ordinary_final_step=91000,
        optimizer_start_step=16000, optimizer_final_step=26000,
        fresh_optimizer=False, coefficient=1.8188207859141674, coefficient_recalibration=False,
        weight_decay=1e-5, gradient_clip=10., features=1323, architecture=[1323,512,512,23],
        context_order='previous_action23_then_incoming_history300',
        first_layer_execution='split_old256_new256_original1000_plus323', export_first_layer_execution='monolithic_float64_1323',
        initial_parity_tolerance_rad=1e-5, initial_byte_gate_required=False, parity_tolerance_rad=1e-5,
        automatic_retry=False, no_checkpoint_selection=True, context_and_normalization_reused=True,
        expansion_performed=False, recovery_rows=1018, recovery_phase_counts=[99,819,100],
        recovery_coefficient=0.2, recovery_objective='equal_three_phase_normalized_MSE',
        response_schedule='unchanged10000_prefix_no_wrap', group_weights=list(GROUP_WEIGHTS),
        consistency_evidence_reviewed=True, learning_rate_values=selected_rates(),
        subjects={'checkpoint': {'sha256':SOURCE_CHECKPOINT}},
        budgets=dict(training_forward_rows=157040000,training_forward_calls=40000,training_updates=10000,
            diagnostic_Torch_rows=1474352,diagnostic_Torch_calls=5764,diagnostic_ORT_rows=368588,diagnostic_ORT_calls=1441,
            calibration_forward_calls=0,calibration_gradient_calls=0,native_calls=0,BFM_calls=0,manual_export_trace_calls=0))

def test_exact_recovery_request():
    recovery_request_identity(request())

@pytest.mark.parametrize('key,value', [
    ('kind','causal_width512_warm_continuation'),('root_selected',False),('updates',3000),
    ('ordinary_start_step',71000),('ordinary_final_step',81000),('optimizer_start_step',0),('optimizer_final_step',16000),
    ('fresh_optimizer',True),('coefficient',1.),('architecture',[1323,256,256,23]),('expansion_performed',True),
    ('recovery_rows',1017),('recovery_phase_counts',[100,819,99]),('recovery_coefficient',1.),
    ('recovery_objective','average_all_rows'),('response_schedule','resampled'),
    ('consistency_evidence_reviewed',False),('context_order','current_history'),('context_and_normalization_reused',False),
    ('parity_tolerance_rad',1e-4),('initial_parity_tolerance_rad',1e-4),('automatic_retry',True)])
def test_wrong_recovery_request_rejected(key,value):
    item=request();item[key]=value
    with pytest.raises(AssertionError):recovery_request_identity(item)

@pytest.mark.parametrize('index',[0,249,250,9999])
def test_rate_schedule_rejects_one_changed_update(index):
    item=request();item['learning_rate_values'][index]*=1.0000001
    with pytest.raises(AssertionError):recovery_request_identity(item)

@pytest.mark.parametrize('key',['training_forward_calls','training_forward_rows','diagnostic_ORT_calls','native_calls'])
def test_changed_budget_rejected(key):
    item=request();item['budgets'][key]+=1
    with pytest.raises(AssertionError):recovery_request_identity(item)

def test_source_checkpoint_rejected():
    item=request();item['subjects']['checkpoint']['sha256']='0'*64
    with pytest.raises(AssertionError):recovery_request_identity(item)

def lineage_fixture():
    from recovery_release import ROLES
    roles=ROLES+('normalization',)
    binding={name:dict(path='E:/synthetic/'+name+'.json',sha256=f'{index+1:064x}') for index,name in enumerate(roles)}
    paths={name:item['path'] for name,item in binding.items()}
    subjects=copy.deepcopy(binding)
    subjects['normalization']['path']='E:/synthetic/source_normalization.npz'
    request_value={'subjects':subjects}
    fit={'direct_subject_sha256':{name:item['sha256'] for name,item in binding.items()},
         'recovery_collection':copy.deepcopy(subjects['collection_report'])}
    pins={item['path']:item['sha256'] for item in binding.values()}
    pins[subjects['normalization']['path']]=subjects['normalization']['sha256']
    source={'collector.py':'f'*64}
    records={
        'collection_report':dict(passed=True,collection_completed=True,rows=1018,control_start=251,
            control_stop_exclusive=1269,fresh_student_state_queries=1,model_fitting_authorized=False,
            request_sha256=binding['collection_request']['sha256'],
            outputs={'expert_rows.npz':binding['recovery_rows']['sha256'],'normalization.npz':binding['normalization']['sha256']},
            input_sha256=copy.deepcopy(pins),source_sha256=source),
        'collection_request':dict(root_selected_collection=True,model_fitting_authorized=False,
            subjects={'qualification':binding['collection_qualification'],'source_review':binding['collection_source_review']},source_sha256=source),
        'collection_qualification':dict(root_authorized_extraction=True,model_fitting_authorized=False,
            control_start=251,control_stop_exclusive=1269,rows=1018),
        'collection_source_review':dict(source_review_pass=True,source_sha256=source),
        'consistency_report':dict(passed=True,evidence_diagnosis_completed=True,old_rows=12958,new_rows=1018,rows=13976,
            input_sha256=copy.deepcopy(pins)),
        'warm_restore_review':dict(source_review_pass=True)}
    return binding,paths,request_value,fit,{paths[key]:value for key,value in records.items()}

def run_lineage(fixture):
    from export_release_gate import consumed_subjects
    from recovery_release import validate_recovery_lineage
    binding,paths,request_value,fit,records=fixture
    validate_recovery_lineage(binding,paths,request_value,fit,read=records.__getitem__,consumed_subjects=consumed_subjects)

def test_lineage_all_exact_original_copied_normalization_allowed():
    run_lineage(lineage_fixture())

@pytest.mark.parametrize('mutation', ['different_request_subject_path','different_report_hash','different_row_output',
    'incomplete_collection','wrong_collection_request','missing_source_review_pin','wrong_source_bytes',
    'incomplete_consistency','missing_recovery_pin','missing_original_normalization_pin','failed_warm_review',
    'extra_student_query'])
def test_lineage_transplants_missing_inputs_and_failed_evidence_rejected(mutation):
    fixture=lineage_fixture();binding,paths,request_value,fit,records=fixture
    collection=records[paths['collection_report']];consistency=records[paths['consistency_report']]
    if mutation=='different_request_subject_path':request_value['subjects']['recovery_rows']['path']='E:/synthetic/unconsumed.json'
    elif mutation=='different_report_hash':fit['direct_subject_sha256']['recovery_rows']='0'*64
    elif mutation=='different_row_output':collection['outputs']['expert_rows.npz']='0'*64
    elif mutation=='incomplete_collection':collection['collection_completed']=False
    elif mutation=='wrong_collection_request':collection['request_sha256']='0'*64
    elif mutation=='missing_source_review_pin':del collection['input_sha256'][paths['collection_source_review']]
    elif mutation=='wrong_source_bytes':records[paths['collection_source_review']]['source_sha256']={'changed.py':'0'*64}
    elif mutation=='incomplete_consistency':consistency['evidence_diagnosis_completed']=False
    elif mutation=='missing_recovery_pin':del consistency['input_sha256'][paths['recovery_rows']]
    elif mutation=='missing_original_normalization_pin':del consistency['input_sha256'][request_value['subjects']['normalization']['path']]
    elif mutation=='failed_warm_review':records[paths['warm_restore_review']]['source_review_pass']=False
    elif mutation=='extra_student_query':collection['fresh_student_state_queries']=2
    with pytest.raises(AssertionError):run_lineage(fixture)
