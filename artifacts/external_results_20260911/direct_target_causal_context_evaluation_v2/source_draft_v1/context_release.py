"""Pure literal release checks for a separately selected matched-study endpoint."""
from export_release_gate import expected_fields, consumed_subjects, direct_subjects

SUBJECTS=('fit_report','checkpoint','head','normalization','training_manifest','training_request','export_manifest',
          'coefficient','source_checkpoint','full_state_generation_request','full_state_generation_report',
          'full_state_data_audit','full_state_data_owner','paired_report','shared_manifest','context_alignment',
          'blinded_fit_report','causal_fit_report')
COEFFICIENT=1.8188207859141674
SOURCE_CHECKPOINT='8a1b67e09285a77910d62dd1b2214c8d3684004e55a82041544e750006b29a0a'


def counter(calls,rows):
    return dict(calls_attempted=calls,calls_returned=calls,calls_synchronized=calls,calls_verified=calls,
                rows_attempted=rows,rows_returned=rows,rows_verified=rows)


def condition_report(report,condition):
    expected_fields(report,dict(completed=True,optimization_completed=True,final_export_diagnostics_completed=True,
        numerical_gate_passed=True,export_parity_passed=True,condition=condition,ordinary_final_step=68000,
        additional_updates=3000,optimizer_step=3000,fresh_optimizer=True,features=1323,context_features=323,
        head_output='normalized_target',fixed_full_state_coefficient=COEFFICIENT,execution_dtype='float64',
        public_input_dtype='float32',public_output_dtype='float32',parity_tolerance_rad=1e-5,
        all_frozen_inputs_unchanged=True,checkpoint_selection=False,native_steps=0,BFM_calls=0))
    assert 0<=report['max_preclip_error_rad']<=1e-5
    counts=report['counts']
    assert counts['training']==counter(9000,44058000)
    assert set(counts['diagnostics'])=={'initial_GPU32','final_GPU32','CPU64','GPU64','ORT64'}
    assert all(value==counter(1437,367570) for value in counts['diagnostics'].values())
    assert all(type(counts[key]) is int and counts[key]==0 for key in
               ('calibration_forward_calls','calibration_gradient_calls','BFM_calls','native_calls','manual_export_trace_calls'))


def validate_release(binding,paths,source,purpose,*,read,sha,bound_file,field,has_hash):
    expected_fields(binding,dict(release_kind='ordinary68000_matched_context_same_weight_fp64_export',
        export_input_dtype='float32',export_output_dtype='float32',export_internal_dtype='float64'))
    condition=binding['context_condition']
    assert condition in ('blinded','causal')
    assert binding['source_checkpoint']['sha256']==SOURCE_CHECKPOINT
    paired=read(paths['paired_report'])
    expected_fields(paired,dict(completed=True,conditions=['blinded','causal'],updates_per_condition=3000,
        ordinary_final_step=68000,fixed_coefficient=COEFFICIENT,
        shared_initial_actor_optimizer_RNG_normalization_exact=True,shared_schedule_exact=True,
        all_frozen_inputs_unchanged=True,checkpoint_selection=False,controller_selected=False,
        context_utility_not_unique_hidden_state_attribution=True))
    for name in ('blinded','causal'):
        role=name+'_fit_report';condition_report(read(paths[role]),name)
        assert paired['condition_report_sha256'][name]==binding[role]['sha256']
    assert binding['fit_report']['sha256']==binding[condition+'_fit_report']['sha256']
    fit=read(paths['fit_report']);condition_report(fit,condition)
    for key,role in [('checkpoint_sha256','checkpoint'),('onnx_sha256','head'),('normalization_sha256','normalization'),
        ('training_request_sha256','training_request'),('frozen_receipt_sha256','training_manifest'),
        ('output_manifest_sha256','export_manifest'),('shared_manifest_sha256','shared_manifest')]:
        assert fit[key]==binding[role]['sha256'],key
    request=read(paths['training_request'])
    expected_fields(request,dict(kind='matched_causal_context_utility_study',root_selected=True,
        conditions=['blinded','causal'],updates_per_condition=3000,ordinary_final_step=68000,
        coefficient=COEFFICIENT,learning_rate=[1e-5,1e-6],context_std_floor=.05,features=1323,
        context_order='previous_action23_then_incoming_history300',coefficient_recalibration=False))
    assert request['subjects']['checkpoint']['sha256']==SOURCE_CHECKPOINT
    assert read(paths['coefficient'])['coefficient']==COEFFICIENT
    alignment=read(paths['context_alignment'])
    expected_fields(alignment,dict(all_nominal_rows_assigned_once=True,nominal_rows=9904,center_rows=3057,
        physical_rows=3054,physical_history_once_shifted_exact=True,physical_actual_prior_exact=True,
        no_model_calls=True,no_native_steps=True))
    generation=read(paths['full_state_generation_report'])
    expected_fields(generation,dict(complete=True,signed_rows=354612,overlap_rows=140622,new_rows=213990))
    assert generation['request_sha256']==binding['full_state_generation_request']['sha256']
    data=read(paths['full_state_data_audit'])
    expected_fields(data,dict(data_review_pass=True,rows_checked=354612,exact_old_overlap_rows=140622,
        independently_recomputed_new_features_and_maps=213990,incompatible_normalized_float32_target_rows=0))
    assert data['request_sha256']==binding['full_state_generation_request']['sha256']
    assert data['generation_report_sha256']==binding['full_state_generation_report']['sha256']
    assert data['corpus_sizes']==dict(nominal=9904,full_state=354612,physical=3054)
    data_owner=read(paths['full_state_data_owner'])
    expected_fields(data_owner,dict(passed=True,data_review_pass=True,raw_exit_known=True,raw_python_exit_code=0,
        exit_code=0,all_current_input_pins_exact=True,processes_absent=True))
    consumed_subjects(data_owner['output_sha256'],{paths['full_state_data_audit']:binding['full_state_data_audit']['sha256']})
    subjects={name:binding[name]['sha256'] for name in SUBJECTS}
    entry=binding['root_training_audit'];audit=read(bound_file(entry))
    assert field(audit,entry['pass_field']) is True
    expected_fields(audit,dict(evidence_audit_passed=True,export_qualified=True))
    consumed_subjects(audit['input_sha256'],{binding[name]['path']:digest for name,digest in subjects.items()})
    owner_entry=binding['fit_owner_completion'];owner=read(bound_file(owner_entry))
    assert field(owner,owner_entry['pass_field']) is True
    expected_fields(owner,dict(raw_exit_known=True,raw_python_exit_code=0,exit_code=0,all_postrun_pins_exact=True,processes_absent=True))
    direct_subjects(owner,subjects)
    release_entry=binding['reviews']['release'];release=read(bound_file(release_entry))
    assert field(release,release_entry['pass_field']) is True
    direct_subjects(release,dict(subjects,root_training_audit=entry['sha256'],fit_owner_completion=owner_entry['sha256']))
    source_entry=binding['reviews']['source'];source_review=read(bound_file(source_entry))
    assert field(source_review,source_entry['pass_field']) is True
    for script in ('evaluation_gate.py','context_release.py','causal_features.py','direct_runtime.py',
                   'head_activation_witness.py' if purpose=='witness' else 'evaluate_direct_target_student.py'):
        assert source_review['source_sha256'].get(script)==sha(source/script),('source_review',script)
    return fit
