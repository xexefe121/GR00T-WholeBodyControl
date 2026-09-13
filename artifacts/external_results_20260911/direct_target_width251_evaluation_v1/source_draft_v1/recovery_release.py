"""Independent metadata checks for the fixed 91000 recovery-trained release."""
import math
from export_release_gate import expected_fields
from balance_contract import GROUP_WEIGHTS

ROLES = ('recovery_rows', 'collection_report', 'collection_request',
         'collection_qualification', 'collection_source_review',
         'consistency_report', 'warm_restore_review')
SOURCE_CHECKPOINT = '825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e'

def selected_rates():
    """Recompute the already selected inclusive ramp and cosine independently."""
    return [1e-6 + (1e-5 - 1e-6) * index / 249 if index < 250 else
            1e-6 + 0.5 * (1e-5 - 1e-6) * (1 + math.cos(math.pi * (index - 250) / 9749))
            for index in range(10000)]

def recovery_request_identity(request):
    expected_fields(request, dict(
        kind='qualified_width251_recovery_warm512_fit', root_selected=True, condition='causal',
        updates=10000, ordinary_start_step=81000, ordinary_final_step=91000,
        optimizer_start_step=16000, optimizer_final_step=26000,
        fresh_optimizer=False, coefficient=1.8188207859141674, coefficient_recalibration=False,
        weight_decay=1e-5, gradient_clip=10., features=1323, architecture=[1323,512,512,23],
        context_order='previous_action23_then_incoming_history300',
        first_layer_execution='split_old256_new256_original1000_plus323',
        export_first_layer_execution='monolithic_float64_1323',
        initial_parity_tolerance_rad=1e-5, initial_byte_gate_required=False, parity_tolerance_rad=1e-5,
        automatic_retry=False, no_checkpoint_selection=True, context_and_normalization_reused=True,
        expansion_performed=False, recovery_rows=1018, recovery_phase_counts=[99,819,100],
        recovery_coefficient=0.2, recovery_objective='equal_three_phase_normalized_MSE',
        response_schedule='unchanged10000_prefix_no_wrap', group_weights=list(GROUP_WEIGHTS),
        consistency_evidence_reviewed=True,
        budgets=dict(training_forward_rows=157040000, training_forward_calls=40000, training_updates=10000,
                     diagnostic_Torch_rows=1474352, diagnostic_Torch_calls=5764,
                     diagnostic_ORT_rows=368588, diagnostic_ORT_calls=1441,
                     calibration_forward_calls=0, calibration_gradient_calls=0,
                     native_calls=0, BFM_calls=0, manual_export_trace_calls=0)))
    actual = request['learning_rate_values']
    assert isinstance(actual, list) and len(actual) == 10000
    assert all(type(value) is float and math.isfinite(value) and value > 0 for value in actual)
    assert actual == selected_rates(), 'Different selected rate schedule'
    assert request['subjects']['checkpoint']['sha256'] == SOURCE_CHECKPOINT

def validate_recovery_lineage(binding, paths, request, fit, *, read, consumed_subjects):
    for role in ROLES:
        item = binding[role]
        consumed_subjects({request['subjects'][role]['path']: request['subjects'][role]['sha256']},
                          {item['path']: item['sha256']})
        assert fit['direct_subject_sha256'][role] == item['sha256'], role
    assert fit['recovery_collection'] == request['subjects']['collection_report']
    collection = read(paths['collection_report'])
    expected_fields(collection, dict(passed=True, collection_completed=True, rows=1018,
                    control_start=251, control_stop_exclusive=1269, fresh_student_state_queries=1,
                    model_fitting_authorized=False))
    assert collection['request_sha256'] == binding['collection_request']['sha256']
    assert collection['outputs']['expert_rows.npz'] == binding['recovery_rows']['sha256']
    assert collection['outputs']['normalization.npz'] == binding['normalization']['sha256']
    selected = read(paths['collection_request'])
    expected_fields(selected, dict(root_selected_collection=True, model_fitting_authorized=False))
    for role, original in [('collection_qualification', 'qualification'), ('collection_source_review', 'source_review')]:
        item = binding[role]
        consumed_subjects({selected['subjects'][original]['path']: selected['subjects'][original]['sha256']},
                          {item['path']: item['sha256']})
        consumed_subjects(collection['input_sha256'], {item['path']: item['sha256']})
    qualification = read(paths['collection_qualification'])
    expected_fields(qualification, dict(root_authorized_extraction=True, model_fitting_authorized=False,
                    control_start=251, control_stop_exclusive=1269, rows=1018))
    source = read(paths['collection_source_review'])
    assert source['source_review_pass'] is True
    assert source['source_sha256'] == collection['source_sha256'] == selected['source_sha256']
    consistency = read(paths['consistency_report'])
    expected_fields(consistency, dict(passed=True, evidence_diagnosis_completed=True,
                    old_rows=12958, new_rows=1018, rows=13976))
    for role in ROLES[:5]:
        item = binding[role]
        consumed_subjects(consistency['input_sha256'], {item['path']: item['sha256']})
    # The fit copies normalization unchanged; consistency consumed the source copy.
    original_normalization = request['subjects']['normalization']
    assert original_normalization['sha256'] == binding['normalization']['sha256']
    consumed_subjects(consistency['input_sha256'],
                      {original_normalization['path']: original_normalization['sha256']})
    assert read(paths['warm_restore_review'])['source_review_pass'] is True
