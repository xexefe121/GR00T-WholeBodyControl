"""Independent literal identities for causal81000 saved execution evidence."""
from saved_common import *

def binding_identity(b,check):
    check.require(b['context_condition']=='causal' and b['architecture']==[1323,512,512,23],'explicit causal1323 activation')
    check.require(b['ordinary_final_step']==81000 and b['release_kind']=='ordinary81000_width512_warm_balanced_context_same_weight_fp64_export','81000 causal width512 release kind')

def warm_identity(fit,energy,check):
    check.require(fit['condition']=='causal' and fit['features']==1323 and fit['context_features']==323,'causal context endpoint')
    check.require(fit['ordinary_start_step']==71000 and fit['ordinary_final_step']==81000 and fit['additional_updates']==10000,'fixed ordinary warm continuation')
    check.require(fit['optimizer_start_step']==6000 and fit['optimizer_step']==16000 and fit['fresh_optimizer'] is False,'retained warm optimizer protocol')
    check.require(fit['architecture']==[1323,512,512,23] and fit['hidden_width']==512 and fit['expansion_seed']==20260912,'selected expanded width and seed')
    check.require(fit['training_first_layer_execution']=='split_old256_new256_original1000_plus323' and fit['export_first_layer_execution']=='monolithic_float64_1323','split training and qualified dense FP64 export')
    check.require(fit['fixed_full_state_coefficient']==1.8188207859141674 and fit['context_and_normalization_reused'] is True,'same coefficient and causal normalization')
    check.require(fit['response_weight_rule']=='mean_six_teacher_group_energies_over_group_energy' and energy['rule']=='Emean/Eg','distinct producer metadata and energy provenance rule')
    check.require(fit['response_group_weights']==energy['group_weights'] and len(energy['group_weights'])==6,'six selected group weights')
    check.require(fit['source_checkpoint_sha256']=='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d','actual causal71000 source')

def intent_identity(intent,report,check):
    check.require(intent['kind']=='independent_recorded_state_intent_inspection' and intent['requested_controls']==1569 and intent['global_start']==0,'completed original intent inspection scope')
    check.require(intent['recorded_controls']==report['attempted_controls'] and intent['intended_segment_completed']==report['full_segment_completed'],'intent actual recorded scope')
    check.require(intent['dynamics_executed'] is False and intent['hardware_authorized'] is False,'saved intent only')
    for name in ('full_lifecycle_source_intent_pass','requested_segment_quiet_pass'):
        check.require(type(intent[name]) is bool,'explicit intent verdict '+name)

def check_release(p,check):
    b=read(p['binding']);launch=read(p['launch']);post=read(p['posthash']);owner=read(p['owner'])
    binding_identity(b,check)
    check.require(launch['binding_sha256']==sha(p['binding']) and post==launch['input_hashes'],'launch binding and complete post map')
    for entry in b['input_files']:contains(post,entry['path'],entry['sha256'])
    check.require(owner['owner_completion_accounting_passed'] is True and owner['mode']=='evaluation' and owner['pins_exact']==len(post),'completed owner accounting')
    start,child,exit_record,raw=[read(p[k]) for k in ('process_start','process_child','process_exit','process_raw')]
    clearance=read(p['clearance']);review=read(p['canonical_review']);absence=owner['process_absence']
    check.require(absence['wrapper_absent'] is True and absence['child_absent'] is True,'owner observed process absence')
    check.require(absence['wrapper_pid']==start['wrapper_pid']==child['wrapper_pid'] and absence['child_pid']==child['child_pid'],'process PID chain')
    check.require(raw['known'] is True and raw['raw_python_exit_code']==owner['raw_python_exit_code']==0,'known producer Python exit zero')
    check.require(exit_record['exit_code']==exit_record['raw_child_exit_code']==owner['diagnostic_exit_code'] and exit_record['exit_code'] in (0,2),'diagnostic versus process outcome')
    check.require(exit_record['error'] is None and exit_record['all_postrun_hashes_exact'] is True,'wrapper completed hash accounting')
    check.require(start['receipt_sha256']==clearance['launch_receipt_sha256']==sha(p['launch']),'concrete launch receipt')
    check.require(start['clearance_sha256']==sha(p['clearance']) and start['review_sha256']==clearance['review']['sha256']==sha(p['canonical_review']),'concrete launch clearance')
    check.require(review['passed'] is True,'canonical source and launch review passed')
    expected={name:b[name]['sha256'] for name in SUBJECTS}
    for name,digest in expected.items():check.require(sha(p['subject_'+name])==digest,'actual release subject '+name)
    root=read(p['root_training_audit']);fit_owner=read(p['fit_owner']);release=read(p['final_release']);fit=read(p['subject_fit_report'])
    for role,entry in [('root_training_audit',b['root_training_audit']),('fit_owner',b['fit_owner_completion']),('final_release',b['reviews']['release']),('source_review',b['reviews']['source'])]:
        value=read(p[role])
        for part in entry['pass_field'].split('.'):value=value[part]
        check.require(value is True and sha(p[role])==entry['sha256'],'literal positive receipt '+role)
    check.require(root['evidence_audit_passed'] is True and root['export_qualified'] is True,'independent training and numerical evidence')
    check.require(fit['completed'] is True and fit['numerical_gate_passed'] is True and fit['export_parity_passed'] is True and fit['ordinary_final_step']==81000,'actual completed81000 fit')
    check.require(fit_owner['raw_exit_known'] is True and fit_owner['raw_python_exit_code']==fit_owner['exit_code']==0 and fit_owner['processes_absent'] is True,'completed fit owner')
    for role,digest in expected.items():
        contains(root['input_sha256'],b[role]['path'],digest)
        check.require(root['direct_subject_sha256'][role]==fit_owner['direct_subject_sha256'][role]==release['direct_subject_sha256'][role]==digest,'literal reviewed release role '+role)
    check.require(release['direct_subject_sha256']['root_training_audit']==sha(p['root_training_audit']) and release['direct_subject_sha256']['fit_owner_completion']==sha(p['fit_owner']),'release root and owner hashes')
    warm_identity(fit,read(p['subject_energy_source']),check)
    source_review=read(p['source_review']);frozen=read(p['frozen'])
    for role,path in p.items():
        if role.startswith('actual_source_'):
            relative=role[len('actual_source_'):];digest=sha(path)
            check.require(digest==frozen['source_sha256'][relative]==source_review['source_sha256'][relative],'actual runtime source '+relative)
    config=read(p['release_configuration']);helper=read(p['helper_review'])
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
    witness=read(p['witness_report']);witness_owner=read(p['witness_owner'])
    check.require(witness['context_condition']=='causal' and witness['features']==1323 and witness['expected_head_calls']==witness['attempted_head_calls']==witness['returned_head_calls']==1,'single causal WSL witness')
    check.require(witness_owner['owner_completion_accounting_passed'] is True and witness_owner['diagnostic_passed'] is True,'completed separate witness')
    check.require(witness['head_sha256']==expected['head'] and witness['physics_steps']==witness['BFM_inference_calls']==0,'witness actual head and zero native/BFM')
    check.require(witness['runtime_binary_sha256'] in post.values() and witness['onnxruntime_version']=='1.23.2' and witness['pinned_WSL_runtime'] is True,'actual pinned WSL ORT binary')
    check.require(witness['intra_op_threads']==witness['inter_op_threads']==1,'witness threads')
    for name in ('main','hold'):
        if name+'_trace' not in p:continue
        trace,report,request=p[name+'_trace'],read(p[name+'_report']),read(p[name+'_request'])
        contains(owner['output_hashes'],trace,sha(trace));contains(owner['output_hashes'],p[name+'_report'],sha(p[name+'_report']))
        check.require(report['trace_sha256']==sha(trace) and report['request_sha256']==sha(p[name+'_request']),'actual '+name+' trace and request')
        check.require(request['head_sha256']==expected['head'] and request['evaluation_binding_sha256']==sha(p['binding']) and request['ordinary_final_step']==81000 and request['context_condition']=='causal',name+' model activation subject')
        check.require(request['physical_hz']==500 and request['control_hz']==50 and request['physical_statewrites_after_initialization']==0 and request['learned_BFM_calls']==0 and request['clock_foundation_connected'] is False and request['hardware_authorized'] is False,name+' unchanged runtime contract')
        physics=read(p[name+'_physics'])
        contains(physics['input_hashes'],trace,sha(trace))
        check.require(physics['recorded_trace_reproduced_through_last_sample'] is True and all(physics['original_trace_comparison'].values()),name+' independent native reproduction')
        check.require(physics['compared_physics_steps']==report['physics_steps'] and physics['private_replay_steps_beyond_recorded_prefix']==0,name+' native exact scope')
    intent=read(p['root_intent'])
    intent_identity(intent,read(p['main_report']),check)
    contains(intent['hashes'],p['main_trace'],sha(p['main_trace']))
    contains(intent['hashes'],p['main_physics'],sha(p['main_physics']))
    return dict(root_intent_sha256=sha(p['root_intent']),direct_subject_sha256=expected,runtime_launch_pins=len(post),actual_runtime_sources=len(frozen['source_sha256']),
                diagnostic_passed=owner['diagnostic_passed'],diagnostic_exit_code=owner['diagnostic_exit_code'])
