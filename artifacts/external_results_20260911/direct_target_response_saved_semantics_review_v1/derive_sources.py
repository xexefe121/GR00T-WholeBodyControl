"""Source-only namespace and literal release adaptation. Never read task arrays."""
from pathlib import Path

BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_context_saved_semantics_review_v1'

def one(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)

def save(name,text):
    with (BASE/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)

def main():
    for name in ('audit_saved.py','context_math.py','fixed_maps.py','test_saved.py','verify_completion.py','test_audit_stage.py','preserved_run_audit_template.ps1.txt'):
        with (BASE/name).open('xb') as f:f.write((OLD/name).read_bytes())
    text=(OLD/'saved_common.py').read_text()
    text=one(text,"RUN=NEW/'direct_target_causal_context_evaluation_v2'","RUN=NEW/'direct_target_causal_response_evaluation_v1'")
    text=one(text,"'paired_report','shared_manifest','context_alignment','blinded_fit_report','causal_fit_report'","'shared_manifest','context_alignment','energy_source'")
    save('saved_common.py',text)
    text=(OLD/'test_context_math.py').read_text().replace('ordinary_final_step=68000','ordinary_final_step=71000').replace('ordinary68000_matched_context_same_weight_fp64_export','ordinary71000_warm_balanced_context_same_weight_fp64_export')
    save('test_context_math.py',text)
    text=(OLD/'release_checks.py').read_text().replace('causal68000','causal71000').replace("==68000","==71000").replace('actual completed68000 fit','actual completed71000 fit').replace('ordinary68000_matched_context_same_weight_fp64_export','ordinary71000_warm_balanced_context_same_weight_fp64_export').replace('68000 causal release kind','71000 causal release kind')
    start=text.index("    paired=read(p['subject_paired_report'])")
    end=text.index("    source_review=read(p['source_review'])",start)
    text=text[:start]+"    warm_identity(fit,read(p['subject_energy_source']),check)\n"+text[end:]
    text=one(text,"fit_owner['direct_subject_sha256'][role]==release['direct_subject_sha256'][role]==digest","root['direct_subject_sha256'][role]==fit_owner['direct_subject_sha256'][role]==release['direct_subject_sha256'][role]==digest")
    text=one(text,"    witness=read(p['witness_report']);witness_owner=read(p['witness_owner'])",'''    config=read(p['release_configuration']);helper=read(p['helper_review'])
    helper_entry=config['launch_helper_review']
    contains(post,p['release_configuration'],sha(p['release_configuration']))
    check.require(config['reviews']==b['reviews'] and config['root_training_audit']==b['root_training_audit'] and config['fit_owner_completion']==b['fit_owner_completion'],'same release configuration')
    positive=helper
    for part in helper_entry['pass_field'].split('.'):positive=positive[part]
    check.require(positive is True and helper_entry['sha256']==sha(p['helper_review'])==release['helper_review_sha256'],'literal positive helper review')
    check.require(release['source_review_sha256']==sha(p['source_review']),'separate runtime source review')
    actual_helpers={role[len('actual_helper_'):]:path for role,path in p.items() if role.startswith('actual_helper_')}
    check.require(set(actual_helpers)==set(helper['helper_sha256']),'complete helper source membership')
    for name,path in actual_helpers.items():
        check.require(sha(path)==helper['helper_sha256'][name],'actual helper source '+name)
        contains(post,path,sha(path))
    audit_owner=read(p['root_audit_owner'])
    check.require(audit_owner['completion_accounting_passed'] is True and audit_owner['evidence_audit_passed'] is True and audit_owner['processes_absent'] is True,'completed independent training audit owner')
    check.require(audit_owner['raw_exit_known'] is True and audit_owner['raw_python_exit_code']==audit_owner['exit_code']==0 and audit_owner['report_sha256']==sha(p['root_training_audit']),'independent training audit known exit and report')
    check.require(release['audit_owner_sha256']==sha(p['root_audit_owner']),'release actual saved audit owner')
    witness=read(p['witness_report']);witness_owner=read(p['witness_owner'])''')
    text=one(text,"    return dict(direct_subject_sha256=expected",'''    intent=read(p['root_intent'])
    intent_identity(intent,read(p['main_report']),check)
    contains(intent['hashes'],p['main_trace'],sha(p['main_trace']))
    contains(intent['hashes'],p['main_physics'],sha(p['main_physics']))
    return dict(root_intent_sha256=sha(p['root_intent']),direct_subject_sha256=expected''')
    save('release_checks.py',text)
    text=(OLD/'prepare_request.py').read_text().replace("==68000","==71000").replace('causal68000_runtime','causal71000_runtime')
    text=one(text,"    parser.add_argument('--hold-physics',type=Path);parser.add_argument('--hold-physics-sha')","    parser.add_argument('--hold-physics',type=Path);parser.add_argument('--hold-physics-sha')\n    parser.add_argument('--root-intent',type=Path,required=True);parser.add_argument('--root-intent-sha',required=True)")
    text=one(text,"    assert sha(args.main_physics)==args.main_physics_sha","    assert sha(args.root_intent)==args.root_intent_sha\n    p['root_intent']=args.root_intent\n    assert sha(args.main_physics)==args.main_physics_sha")
    text=one(text,"    p['audit_source_review']=args.source_review",'''    p['audit_source_review']=args.source_review
    p['release_configuration']=RUN/'release_reviews.json'
    config=read(p['release_configuration']);helper_entry=config['launch_helper_review']
    p['helper_review']=local(helper_entry['path']);assert sha(p['helper_review'])==helper_entry['sha256']
    for name in read(p['helper_review'])['helper_sha256']:p['actual_helper_'+name]=RUN/name
    p['root_audit_owner']=p['root_training_audit'].parent.parent/'owner_completion.json'
    release=read(p['final_release']);assert sha(p['root_audit_owner'])==release['audit_owner_sha256']''')
    save('prepare_request.py',text)

if __name__=='__main__':main()
