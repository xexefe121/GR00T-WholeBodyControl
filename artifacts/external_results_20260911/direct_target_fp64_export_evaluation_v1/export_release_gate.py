"""Pure release validation; no model, optimizer, native or dependency installation."""
import re
from pathlib import Path

def validate_release(binding,paths,source,purpose,*,read,sha,bound_file,field,has_hash):
    """Require independent training evidence and a new, separately qualified export.

    The original fit record is deliberately kept failed. Its checkpoint can be
    released only through this new export chain, not by changing the old flags.
    """
    fit=read(paths['fit_report']);export=read(paths['export_report'])
    assert binding['release_kind']=='same55000_checkpoint_fp64_export'
    assert binding['export_input_dtype']=='float32' and binding['export_output_dtype']=='float32'
    assert binding['export_internal_dtype']=='float64'
    assert fit['ordinary_final_step']==55000 and fit['additional_updates']==50000
    assert fit['optimization_completed'] is True and fit['final_export_diagnostics_completed'] is True
    assert fit['completed'] is False and fit['numerical_gate_passed'] is False and fit['export_parity_passed'] is False
    assert fit['checkpoint_sha256']==binding['checkpoint']['sha256']
    assert fit['onnx_sha256']==binding['source_head']['sha256']
    assert fit['features']==1000 and fit['head_output']=='normalized_target'
    assert export['ordinary_final_step']==55000 and export['optimizer_updates']==0
    assert export['features']==1000 and export['head_output']=='normalized_target'
    for key in ('completed','numerical_gate_passed','export_parity_passed'):assert export[key] is True,key
    for key,role in (('checkpoint_sha256','checkpoint'),('source_fit_report_sha256','fit_report'),
                     ('source_onnx_sha256','source_head'),('normalization_sha256','normalization'),('onnx_sha256','head')):
        assert export[key]==binding[role]['sha256'],key
    assert export['parity_tolerance_rad']==1e-5
    assert 0<=export['max_preclip_error_rad']<=export['parity_tolerance_rad']
    assert export['public_input_dtype']==export['public_output_dtype']=='float32' and export['execution_dtype']=='float64'
    assert export['ELU_implementation']=='Where(x>0,x,Exp(Min(x,0))-1)'
    assert export['optimization_completed_source'] is True
    assert export['BFM_calls']==export['native_steps']==0
    assert export['checkpoint_selection'] is False and export['all_frozen_inputs_unchanged'] is True
    # These are directly named immutable subjects, never recursively traversed
    # training corpora or a loose digest found somewhere in an unrelated report.
    def review(role,subjects):
        entry=binding[role] if role in ('root_training_audit','export_owner_completion') else binding['reviews'][role]
        receipt=read(bound_file(entry));assert field(receipt,entry['pass_field']) is True,role
        direct=receipt['direct_subject_sha256']
        assert isinstance(direct,dict) and subjects
        for name,digest in subjects.items():
            assert re.fullmatch('[0-9a-f]{64}',digest) and direct.get(name)==digest,(role,name)
        return receipt
    original={k:binding[k]['sha256'] for k in ('fit_report','checkpoint','source_head','normalization','training_manifest','training_request')}
    review('root_training_audit',original)
    review('dataset',{'training_manifest':original['training_manifest']})
    review('training',dict(original,root_training_audit=binding['root_training_audit']['sha256']))
    new={k:binding[k]['sha256'] for k in ('head','checkpoint','fit_report','source_head','normalization','export_report','export_request','export_manifest')}
    owner=review('export_owner_completion',new)
    assert owner['raw_exit_known'] is True and owner['raw_python_exit_code']==owner['exit_code']==0
    assert owner['all_postrun_pins_exact'] is True and owner['processes_absent'] is True
    review('export',dict(new,root_training_audit=binding['root_training_audit']['sha256'],
                         export_owner_completion=binding['export_owner_completion']['sha256']))
    source_entry=binding['reviews']['source'];source_review=read(bound_file(source_entry))
    assert field(source_review,source_entry['pass_field']) is True
    script=source/('head_activation_witness.py' if purpose=='witness' else 'evaluate_direct_target_student.py')
    # Existing source-review receipts already expose full pinned module hashes.
    assert has_hash(source_review,sha(script)) and has_hash(source_review,sha(Path(__file__)))
    return fit
