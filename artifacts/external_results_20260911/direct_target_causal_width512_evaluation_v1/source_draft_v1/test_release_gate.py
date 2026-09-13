"""Literal fake receipt chains only. No task models, weights or physics."""
import copy
import hashlib
from pathlib import Path
import pytest
from export_release_gate import SUBJECTS,validate_release,path_key


def fixture():
    def digest(name):return hashlib.sha256(name.encode()).hexdigest()
    names=list(SUBJECTS)+['root_training_audit','fit_owner_completion','source_review','release_review']
    entries={n:dict(path='E:/synthetic/'+n+'.json',sha256=digest(n),pass_field='passed') for n in names}
    binding=dict(entries,release_kind='ordinary65000_full58_same_weight_fp64_export',export_input_dtype='float32',
        export_output_dtype='float32',export_internal_dtype='float64',reviews=dict(source=entries['source_review'],release=entries['release_review']))
    paths={n:entries[n]['path'] for n in SUBJECTS}
    fit=dict(ordinary_final_step=65000,additional_updates=10000,optimizer_step=10000,restored_start_step=55000,
        fresh_optimizer=True,completed=True,optimization_completed=True,final_export_diagnostics_completed=True,
        numerical_gate_passed=True,export_parity_passed=True,features=1000,head_output='normalized_target',
        nominal_rows=9904,full_state_endpoint_rows=354612,physical_rows=3054,full_state_cells=54,
        execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',
        ELU_implementation='Where(x>0,x,Exp(Min(x,0))-1)',parity_tolerance_rad=1e-5,max_preclip_error_rad=0.,
        full_state_coefficient=7.,BFM_calls=0,native_steps=0,all_frozen_inputs_unchanged=True,
        checkpoint_selection=False,FP32_ONNX_release=False,hardware_authorized=False)
    for key,role in [('checkpoint_sha256','checkpoint'),('onnx_sha256','head'),('normalization_sha256','normalization'),
        ('training_request_sha256','training_request'),('frozen_receipt_sha256','training_manifest'),
        ('output_manifest_sha256','export_manifest'),('coefficient_sha256','coefficient')]:fit[key]=entries[role]['sha256']
    coefficient=dict(coefficient=7.,optimizer_updates=0,schedule_index=0,model_unchanged=True,RNG_unchanged=True,
        counters=dict(attempted=3,returned=3,synchronized=3,verified=3))
    generation=dict(complete=True,signed_rows=354612,overlap_rows=140622,new_rows=213990,
        request_sha256=entries['full_state_generation_request']['sha256'])
    data=dict(data_review_pass=True,rows_checked=354612,exact_old_overlap_rows=140622,
        independently_recomputed_new_features_and_maps=213990,incompatible_normalized_float32_target_rows=0,
        request_sha256=entries['full_state_generation_request']['sha256'],generation_report_sha256=entries['full_state_generation_report']['sha256'],
        corpus_sizes=dict(nominal=9904,full_state=354612,physical=3054))
    data_owner=dict(passed=True,data_review_pass=True,raw_exit_known=True,raw_python_exit_code=0,exit_code=0,
        all_current_input_pins_exact=True,processes_absent=True,
        output_sha256={'/mnt/e/synthetic/full_state_data_audit.json':entries['full_state_data_audit']['sha256']})
    subjects={n:entries[n]['sha256'] for n in SUBJECTS}
    root=dict(passed=True,evidence_audit_passed=True,export_qualified=True,
        input_sha256={'/mnt/e/synthetic/'+n+'.json':entries[n]['sha256'] for n in SUBJECTS})
    owner=dict(passed=True,raw_exit_known=True,raw_python_exit_code=0,exit_code=0,all_postrun_pins_exact=True,
        processes_absent=True,direct_subject_sha256=subjects.copy())
    release=dict(passed=True,direct_subject_sha256=dict(subjects,root_training_audit=entries['root_training_audit']['sha256'],
        fit_owner_completion=entries['fit_owner_completion']['sha256']))
    source=Path('E:/synthetic_source')
    scripts=('evaluation_gate.py','export_release_gate.py','head_activation_witness.py','evaluate_direct_target_student.py')
    source_review=dict(passed=True,source_sha256={n:digest(n) for n in scripts})
    records={paths['fit_report']:fit,paths['coefficient']:coefficient,paths['full_state_generation_report']:generation,
        paths['full_state_data_audit']:data,paths['full_state_data_owner']:data_owner,
        entries['root_training_audit']['path']:root,entries['fit_owner_completion']['path']:owner,
        entries['release_review']['path']:release,entries['source_review']['path']:source_review}
    context=dict(read=lambda p:records[str(p)],sha=lambda p:digest(Path(p).name),bound_file=lambda e:e['path'],
        field=lambda r,k:r[k],has_hash=lambda r,h:False)
    return binding,paths,source,records,context


@pytest.mark.parametrize('purpose',['witness','evaluation'])
def test_complete_independent_release(purpose):
    binding,paths,source,records,context=fixture()
    assert validate_release(binding,paths,source,purpose,**context) is records[paths['fit_report']]


@pytest.mark.parametrize('role,key,value',[
    ('fit_report','ordinary_final_step',55000),('fit_report','fresh_optimizer',False),
    ('fit_report','completed',False),('fit_report','export_parity_passed',False),
    ('fit_report','max_preclip_error_rad',1.00001e-5),('fit_report','full_state_coefficient',float('nan')),
    ('fit_report','full_state_coefficient',-1.),('fit_report','full_state_cells',9),
    ('fit_report','native_steps',False),('fit_report','FP32_ONNX_release',True),
    ('coefficient','optimizer_updates',1),('coefficient','RNG_unchanged',False),
    ('full_state_generation_report','new_rows',213989),('full_state_data_audit','incompatible_normalized_float32_target_rows',1),
    ('full_state_data_owner','all_current_input_pins_exact',False),
    ('root_training_audit','export_qualified',False),('fit_owner_completion','processes_absent',False),
])
def test_failed_or_wrong_receipt_rejected(role,key,value):
    binding,paths,source,records,context=fixture()
    records[binding[role]['path']][key]=value
    with pytest.raises(AssertionError):validate_release(binding,paths,source,'witness',**context)


@pytest.mark.parametrize('role,map_name,key',[
    ('fit_owner_completion','direct_subject_sha256','source_checkpoint'),
    ('release_review','direct_subject_sha256','root_training_audit'),
    ('source_review','source_sha256','export_release_gate.py'),
    ('root_training_audit','input_sha256','/mnt/e/synthetic/full_state_data_owner.json'),
    ('full_state_data_owner','output_sha256','/mnt/e/synthetic/full_state_data_audit.json'),
])
def test_missing_literal_subject_rejected(role,map_name,key):
    binding,paths,source,records,context=fixture()
    records[binding[role]['path']][map_name].pop(key)
    with pytest.raises(AssertionError):validate_release(binding,paths,source,'evaluation',**context)


def test_cross_platform_path_keys_preserve_identity():
    assert path_key('/mnt/e/synthetic/report.json')==path_key('E:\\synthetic\\report.json')
    assert path_key('/mnt/e/synthetic/report.json')!=path_key('E:/different/report.json')
