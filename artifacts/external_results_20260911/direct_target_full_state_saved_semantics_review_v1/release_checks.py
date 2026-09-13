"""Independent literal identities for unified65000 saved execution evidence."""
from saved_common import *

def check_release(p,check):
    b=read(p['binding']);launch=read(p['launch']);post=read(p['posthash']);owner=read(p['owner'])
    check.require(b['ordinary_final_step']==65000 and b['release_kind']=='ordinary65000_full58_same_weight_fp64_export','65000 release kind')
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
    check.require(fit['completed'] is True and fit['numerical_gate_passed'] is True and fit['export_parity_passed'] is True and fit['ordinary_final_step']==65000,'actual completed65000 fit')
    check.require(fit_owner['raw_exit_known'] is True and fit_owner['raw_python_exit_code']==fit_owner['exit_code']==0 and fit_owner['processes_absent'] is True,'completed fit owner')
    for role,digest in expected.items():
        contains(root['input_sha256'],b[role]['path'],digest)
        check.require(fit_owner['direct_subject_sha256'][role]==release['direct_subject_sha256'][role]==digest,'literal reviewed release role '+role)
    check.require(release['direct_subject_sha256']['root_training_audit']==sha(p['root_training_audit']) and release['direct_subject_sha256']['fit_owner_completion']==sha(p['fit_owner']),'release root and owner hashes')
    source_review=read(p['source_review']);frozen=read(p['frozen'])
    for role,path in p.items():
        if role.startswith('actual_source_'):
            relative=role[len('actual_source_'):];digest=sha(path)
            check.require(digest==frozen['source_sha256'][relative]==source_review['source_sha256'][relative],'actual runtime source '+relative)
    witness=read(p['witness_report']);witness_owner=read(p['witness_owner'])
    check.require(witness_owner['owner_completion_accounting_passed'] is True and witness_owner['diagnostic_passed'] is True,'completed separate witness')
    check.require(witness['head_sha256']==expected['head'] and witness['physics_steps']==witness['BFM_inference_calls']==0,'witness actual head and zero native/BFM')
    check.require(witness['runtime_binary_sha256'] in post.values() and witness['onnxruntime_version']=='1.23.2' and witness['pinned_WSL_runtime'] is True,'actual pinned WSL ORT binary')
    check.require(witness['intra_op_threads']==witness['inter_op_threads']==1,'witness threads')
    for name in ('main','hold'):
        if name+'_trace' not in p:continue
        trace,report,request=p[name+'_trace'],read(p[name+'_report']),read(p[name+'_request'])
        contains(owner['output_hashes'],trace,sha(trace));contains(owner['output_hashes'],p[name+'_report'],sha(p[name+'_report']))
        check.require(report['trace_sha256']==sha(trace) and report['request_sha256']==sha(p[name+'_request']),'actual '+name+' trace and request')
        check.require(request['head_sha256']==expected['head'] and request['evaluation_binding_sha256']==sha(p['binding']) and request['ordinary_final_step']==65000,name+' model activation subject')
        check.require(request['physical_hz']==500 and request['control_hz']==50 and request['physical_statewrites_after_initialization']==0 and request['learned_BFM_calls']==0 and request['clock_foundation_connected'] is False and request['hardware_authorized'] is False,name+' unchanged runtime contract')
        physics=read(p[name+'_physics'])
        contains(physics['input_hashes'],trace,sha(trace))
        check.require(physics['recorded_trace_reproduced_through_last_sample'] is True and all(physics['original_trace_comparison'].values()),name+' independent native reproduction')
        check.require(physics['compared_physics_steps']==report['physics_steps'] and physics['private_replay_steps_beyond_recorded_prefix']==0,name+' native exact scope')
    return dict(direct_subject_sha256=expected,runtime_launch_pins=len(post),actual_runtime_sources=len(frozen['source_sha256']),
                diagnostic_passed=owner['diagnostic_passed'],diagnostic_exit_code=owner['diagnostic_exit_code'])
