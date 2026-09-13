"""Derive eight helpers only; root-owned runtime source remains untouched."""
from pathlib import Path
import json,hashlib,ast
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_context_evaluation_v2'
NAMES=('runtime_inventory.py','freeze_final_package.py','prepare_review_configuration.py','prepare_bound_launcher.py',
       'diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py','test_release_helpers.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
runtime_before={p.relative_to(BASE/'source_draft_v1').as_posix():sha(p) for p in (BASE/'source_draft_v1').rglob('*.py')}
old_map={};new_map={};changes={}
for name in NAMES:
    text=(OLD/name).read_text(encoding='utf-8');original=text
    if name=='freeze_final_package.py':
        text=text.replace('context68000','warm balanced71000').replace("FIT=NEW/'direct_target_causal_context_study_v2'","FIT=NEW/'direct_target_causal_response_balanced_student_v1'")
        text=text.replace("condition not in ('blinded','causal')","condition!='causal'").replace('An explicitly selected blinded or causal condition is required.','Only the explicitly selected causal continuation condition is allowed.')
        text=text.replace("dest=FIT/'fit'/condition","dest=FIT/'fit'")
        text=text.replace("        paired_report=FIT/'fit/paired_report.json',shared_manifest=shared/'output_manifest.json',\n        context_alignment=shared/'context_alignment.json',\n        blinded_fit_report=FIT/'fit/blinded/report.json',causal_fit_report=FIT/'fit/causal/report.json',",
                          "        shared_manifest=shared/'output_manifest.json',context_alignment=shared/'context_alignment.json',\n        energy_source=Path(request['subjects']['energy_source']['path']),")
        text=text.replace('ordinary_final_step=68000','ordinary_final_step=71000')
        text=text.replace('ordinary68000_matched_context_same_weight_fp64_export','ordinary71000_warm_balanced_context_same_weight_fp64_export')
        text=text.replace('direct_absolute_target_1323_causal_context_study','direct_absolute_target_1323_causal_response_balanced')
        text=text.replace('one_ordinary68000_context_canonical_evaluation','one_ordinary71000_response_balanced_canonical_evaluation')
    elif name=='prepare_review_configuration.py':
        text=text.replace('context68000','warm balanced71000').replace("choices=('blinded','causal')","choices=('causal',)")
    elif name=='runtime_inventory.py':
        text=text.replace('reviewed68000 context runtime source','reviewed71000 response-balanced context runtime source').replace('explicit_context68000_fixed_runtime_inventory','explicit_response71000_fixed_runtime_inventory')
    elif name=='test_release_helpers.py':
        text=text.replace("[None,'','both','best','CAUSAL',0,False]","[None,'','both','best','blinded','CAUSAL',0,False]")
        text=text.replace("@pytest.mark.parametrize('condition',['blinded','causal'])","@pytest.mark.parametrize('condition',['causal'])")
        text=text.replace('test_exact_condition_path_and_all18_roles','test_exact_condition_path_and_all16_roles')
        text=text.replace("('checkpoint','coefficient_source','full_state_root_audit','full_state_root_owner')","('checkpoint','coefficient_source','full_state_root_audit','full_state_root_owner','energy_source')")
        text=text.replace('len(SUBJECTS)==18','len(SUBJECTS)==16').replace("tmp_path/'fit'/condition/'student_head.onnx'","tmp_path/'fit/student_head.onnx'")
        text=text.replace("    assert paths['fit_report']==paths[condition+'_fit_report']","    assert paths['fit_report']==tmp_path/'fit/report.json'\n    assert paths['energy_source']==tmp_path/'energy_source'\n    assert not any(name in paths for name in ('paired_report','blinded_fit_report','causal_fit_report'))")
        text=text.replace("b['ordinary_final_step']==68000","b['ordinary_final_step']==71000")
    ast.parse(text,filename=name)
    with (BASE/name).open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    old_map[name]=sha(OLD/name);new_map[name]=sha(BASE/name);changes[name]=text!=original
runtime_after={p.relative_to(BASE/'source_draft_v1').as_posix():sha(p) for p in (BASE/'source_draft_v1').rglob('*.py')}
assert runtime_after==runtime_before
receipt=dict(preparation_only=True,helper_sha256=new_map,baseline_helper_sha256=old_map,
    changed_helper_files=[k for k,v in changes.items() if v],unchanged_helper_files=[k for k,v in changes.items() if not v],
    source_sha256=runtime_after,all37_root_runtime_sources_unchanged=True,
    protocol='Single causal71000 fit root and16 release roles; original1569+conditional250, one separate WSL witness.',
    condition_default=None,actual_binding_created=False,model_calls=0,native_steps=0,
    derivation_source_sha256=sha(__file__))
with (BASE/'launch_helper_derivation.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(dict(helpers=len(new_map),runtime_sources=len(runtime_after),changed=receipt['changed_helper_files'])))
