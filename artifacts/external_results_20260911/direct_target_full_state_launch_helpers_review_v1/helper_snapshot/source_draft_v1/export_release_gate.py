"""Unified ordinary65000 full-state fit release. Pure immutable receipt checks."""
import math
import re

SUBJECTS=('fit_report','checkpoint','head','normalization','training_manifest','training_request','export_manifest','coefficient',
          'source_checkpoint','full_state_generation_request','full_state_generation_report','full_state_data_audit','full_state_data_owner')

def path_key(value):
    text=str(value).replace('\\','/')
    if text.startswith('/mnt/') and text[6:7]=='/':text=text[5]+':'+text[6:]
    return text.casefold()

def expected_fields(actual,expected):
    for key,value in expected.items():
        assert type(actual[key]) is type(value) and actual[key]==value,('literal_field',key)

def direct_subjects(receipt,expected):
    actual=receipt['direct_subject_sha256']
    assert isinstance(actual,dict) and expected
    for name,digest in expected.items():
        assert re.fullmatch('[0-9a-f]{64}',digest) and actual.get(name)==digest,('direct_subject',name)

def consumed_subjects(receipt,expected):
    actual={}
    for path,digest in receipt.items():
        key=path_key(path)
        assert key not in actual or actual[key]==digest,'Ambiguous consumed path'
        actual[key]=digest
    for path,digest in expected.items():
        assert actual.get(path_key(path))==digest,('consumed_subject',str(path))

def validate_release(binding,paths,source,purpose,*,read,sha,bound_file,field,has_hash):
    expected_fields(binding,dict(release_kind='ordinary65000_full58_same_weight_fp64_export',
        export_input_dtype='float32',export_output_dtype='float32',export_internal_dtype='float64'))
    fit=read(paths['fit_report'])
    expected_fields(fit,dict(ordinary_final_step=65000,additional_updates=10000,optimizer_step=10000,restored_start_step=55000,
        fresh_optimizer=True,completed=True,optimization_completed=True,final_export_diagnostics_completed=True,
        numerical_gate_passed=True,export_parity_passed=True,features=1000,head_output='normalized_target',
        nominal_rows=9904,full_state_endpoint_rows=354612,physical_rows=3054,full_state_cells=54,
        execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',
        ELU_implementation='Where(x>0,x,Exp(Min(x,0))-1)',parity_tolerance_rad=1e-5,
        BFM_calls=0,native_steps=0,all_frozen_inputs_unchanged=True,checkpoint_selection=False,FP32_ONNX_release=False,hardware_authorized=False))
    assert 0<=fit['max_preclip_error_rad']<=1e-5
    assert math.isfinite(fit['full_state_coefficient']) and fit['full_state_coefficient']>0
    for key,role in [('checkpoint_sha256','checkpoint'),('onnx_sha256','head'),('normalization_sha256','normalization'),
        ('training_request_sha256','training_request'),('frozen_receipt_sha256','training_manifest'),
        ('output_manifest_sha256','export_manifest'),('coefficient_sha256','coefficient')]:
        assert fit[key]==binding[role]['sha256'],key
    coefficient=read(paths['coefficient'])
    assert coefficient['coefficient']==fit['full_state_coefficient']
    expected_fields(coefficient,dict(optimizer_updates=0,schedule_index=0,model_unchanged=True,RNG_unchanged=True))
    assert coefficient['counters']==dict(attempted=3,returned=3,synchronized=3,verified=3)
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
    entry=binding['root_training_audit'];root=read(bound_file(entry))
    assert field(root,entry['pass_field']) is True
    expected_fields(root,dict(evidence_audit_passed=True,export_qualified=True))
    consumed_subjects(root['input_sha256'],{binding[name]['path']:digest for name,digest in subjects.items()})
    owner_entry=binding['fit_owner_completion'];owner=read(bound_file(owner_entry))
    assert field(owner,owner_entry['pass_field']) is True
    expected_fields(owner,dict(raw_exit_known=True,raw_python_exit_code=0,exit_code=0,all_postrun_pins_exact=True,processes_absent=True))
    direct_subjects(owner,subjects)
    release_entry=binding['reviews']['release'];release=read(bound_file(release_entry))
    assert field(release,release_entry['pass_field']) is True
    direct_subjects(release,dict(subjects,root_training_audit=entry['sha256'],fit_owner_completion=owner_entry['sha256']))
    source_entry=binding['reviews']['source'];source_review=read(bound_file(source_entry))
    assert field(source_review,source_entry['pass_field']) is True
    for script in ('evaluation_gate.py','export_release_gate.py',
                   'head_activation_witness.py' if purpose=='witness' else 'evaluate_direct_target_student.py'):
        assert source_review['source_sha256'].get(script)==sha(source/script),('source_review',script)
    return fit
