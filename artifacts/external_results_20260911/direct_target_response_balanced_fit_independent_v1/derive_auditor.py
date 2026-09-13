"""Derive one saved-only warm audit; never executes it."""
from pathlib import Path
import difflib,hashlib,json
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
OLD=NEW/'direct_target_context_pair_fit_independent_v1/source_draft_v3';DEST=BASE/'source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=json.loads((NEW/'direct_target_context_pair_audit_root_review_v3/review.json').read_text())
for name in ('audit_saved_pair.py','audit_math.py','audit_restoration.py','audit_graph.py','audit_context_math.py','audit_full_state_math.py'):
    assert sha(OLD/name)==root['source_sha256'][name]
for name in ('audit_math.py','audit_restoration.py','audit_graph.py','audit_context_math.py','audit_full_state_math.py'):(DEST/name).write_bytes((OLD/name).read_bytes())
original=(OLD/'audit_saved_pair.py').read_text();text=original;changes=[]
def replace(old,new):
    global text
    assert text.count(old)==1,(old[:100],text.count(old))
    text=text.replace(old,new);changes.append(old[:120])
def block(start,end,new):
    global text
    a=text.index(start);b=text.index(end,a);changes.append(text[a:b][:120]);text=text[:a]+new+text[b:]
text=text.replace('68000','71000').replace('65000','68000')
text=text.replace('fit_process_v2','fit_process_v1')
replace('"""Audit saved paired-context evidence. No forwards, gradients, optimizers or physics."""','"""Audit saved warm71000 evidence. No forwards, gradients, optimizers or physics."""')
replace('from audit_graph import audit_graph','from audit_graph import audit_graph\nfrom audit_balanced_math import energy_weights,balanced_metrics,warm_initial_errors,ledger_expectations\nfrom audit_release import release_paths')
replace("ar['kind']=='saved_matched_context_pair_only'","ar['kind']=='saved_response_balanced_warm_only'")
replace("        for role,subject in ar['subjects'].items():bind(subject['path'],subject['sha256'])", "        writer=ar['request_writer'];bind(writer['path'],writer['sha256'])\n        check('reviewed_request_writer',Path(writer['path']).name=='prepare_audit_request.py' and review['helper_sha256']['prepare_audit_request.py']==writer['sha256'])\n        for role,subject in ar['subjects'].items():bind(subject['path'],subject['sha256'])")
replace("        for role,path in [('training_request',req_path),('frozen_inputs',frozen_path),('launcher',clearance['launcher_path'])]:literal(launch_review['subjects'][role],path,role)", "        for field,digest in dict(training_request_sha256=training_sha,frozen_receipt_sha256=frozen_sha,launcher_sha256=clearance['launcher_sha256']).items():check('concrete_literal:'+field,launch_review[field]==digest)\n        compare('concrete_source_map',launch_review['source_sha256'],frozen['source_sha256'])")
replace("        stage='shared_data_and_context'", "        frozen_members={canonical(p):d for p,d in frozen['input_sha256'].items()}\n        for group in ('paths','full_state_paths','restoration_predictions','context_paths'):\n            for name,path in request[group].items():check('consumed_path_membership:'+group+'/'+name,frozen_members.get(canonical(path))==sha(bind(path)))\n        stage='shared_data_and_context'")
replace("dict(**request,condition=condition,initialization_sha256=", "dict(request,condition=condition,initialization_sha256=")
block('        expected_budget=dict(',"        clearance=read(",'''        expected_budget=dict(training_forward_rows=44058000,training_forward_calls=9000,training_updates=3000,
            diagnostic_Torch_rows=1470280,diagnostic_Torch_calls=5748,diagnostic_ORT_rows=367570,diagnostic_ORT_calls=1437,
            calibration_forward_calls=0,calibration_gradient_calls=0,native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
        for name,value in dict(kind='causal_response_balanced_continuation',root_selected=True,condition='causal',
            updates=3000,ordinary_start_step=68000,ordinary_final_step=71000,optimizer_start_step=3000,optimizer_final_step=6000,
            fresh_optimizer=False,coefficient=WEIGHT,budgets=expected_budget,learning_rate=[1e-5,1e-6],weight_decay=1e-5,
            gradient_clip=10.,coefficient_recalibration=False,initial_byte_gate_required=False,
            initial_parity_tolerance_rad=1e-5,parity_tolerance_rad=1e-5).items():compare('request.'+name,request[name],value)
''')
replace("compare('context_source_map',alignment['source_input_sha256'],frozen['input_sha256'])", "old_frozen=read(bind(request['subjects']['source_frozen_inputs']['path']))\n        compare('context_source_map',alignment['source_input_sha256'],old_frozen['input_sha256'])\n        for key,path in request['context_paths'].items():\n            name='source_output_manifest.json' if key=='manifest' else Path(path).name\n            check('shared_byte_copy:'+key,sha(bind(shared/name))==sha(bind(path)))")
replace("exact('original_'+name,norm['original_'+name],source_norm[name]);exact('combined_original_'+name,norm[name][:1000],source_norm[name])", "exact('reused_full_'+name,norm[name],source_norm[name]);exact('combined_original_'+name,norm[name][:1000],source_norm['original_'+name])")
replace("old=npy(request['schedule_paths'][key]);check('old_schedule_shape:'+key,old.shape==(10000,864));exact('old_prefix:'+key,old[:3000],value);exact('shared_schedule:'+key,npy(shared/('schedule_'+key+'.npy')),value)", "old=npy(request['context_paths']['schedule_'+key]);check('old_schedule_shape:'+key,old.shape==(3000,864));exact('reused_schedule:'+key,old,value);exact('shared_schedule:'+key,npy(shared/('schedule_'+key+'.npy')),value)")
replace("'8a1b67e09285a77910d62dd1b2214c8d3684004e55a82041544e750006b29a0a'", "'10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd'")
replace("        calculated={};initializations={};qualification={};initial_outputs={};graph_reports={};report_hashes={}\n        for condition in ('blinded','causal'):\n            stage=condition;folder=fit/condition", "        energy=read(bind(request['subjects']['energy_source']['path']));group_weights=energy_weights(energy)\n        for path,digest in energy['input_sha256'].items():bind(path,digest)\n        calculated={};initializations={};qualification={};initial_outputs={};graph_reports={};report_hashes={}\n        for condition in ('causal',):\n            stage=condition;folder=fit")
replace("'physical_cell_losses.npy','diagnostic_calls.jsonl'", "'physical_cell_losses.npy','original_objectives.npy','balanced_full_state_cell_losses.npy','diagnostic_calls.jsonl'")
replace("            if condition=='causal':required.append('initial_comparison.json')\n",'')
replace("optimizer_step=3000,fresh_optimizer=True,features=1323", "optimizer_step=6000,fresh_optimizer=False,features=1323")
replace("            compare(condition+'_carried_request'", "            for field,value in dict(ordinary_start_step=68000,optimizer_start_step=3000,context_and_normalization_reused=True,source_checkpoint_sha256=request['subjects']['checkpoint']['sha256'],energy_source_sha256=request['subjects']['energy_source']['sha256']).items():compare('report_extra.'+field,report[field],value)\n            compare(condition+'_carried_request'")
block('            for name in names:\n',"            for name in ('feature_mean','feature_std'):",'''            check('warm_initial_actor_optimizer_RNG',not warm_initial_errors(init,source))
            for key,value in dict(ordinary_start_step=68000,optimizer_start_step=3000,condition=condition,source_checkpoint_sha256=request['subjects']['checkpoint']['sha256']).items():compare(condition+'_init.'+key,init[key],value)
            tree(condition+'_final_rng',checkpoint['rng'],source['rng'])
''')
replace("optimizer_step=3000,fresh_optimizer=True,context_order=", "optimizer_step=6000,fresh_optimizer=False,context_order=")
replace("check(condition+'_optimizer_step:'+str(index),float(state['step'])==3000)", "check(condition+'_optimizer_step:'+str(index),state['step'].dtype==torch.float32 and state['step'].shape==torch.Size([]) and float(state['step'])==6000)")
replace("[('initial',init['optimizer_state'],1e-5),('final',optimizer,1e-6)]", "[('initial',init['optimizer_state'],1e-6),('final',optimizer,1e-6)]")
block("            if condition=='blinded':", "            optimization=", "            exact('checkpoint_group_weights',np.asarray(checkpoint['response_group_weights'],np.float64),group_weights)\n            check('checkpoint_weight_rule',checkpoint['response_weight_rule']=='Emean/Eg')\n")
replace("optimization['optimizer_step']==3000", "optimization['optimizer_step']==6000")
block('            for index,(name,width) in enumerate(zip(CORPORA,(15,54,9))):', "            outputs=",'''            cell_values={}
            for name,width in zip(CORPORA,(15,54,9)):
                cells=npy(folder/(name+'_cell_losses.npy'));check(condition+'_cell_shape:'+name,cells.shape==(3000,width) and cells.dtype==np.float64 and np.isfinite(cells).all() and np.all(cells>=0));cell_values[name]=cells
            weighted_cells=npy(folder/'balanced_full_state_cell_losses.npy')
            wanted=ledger_expectations(cell_values['nominal'],cell_values['full_state'],cell_values['physical'],group_weights)
            exact('balanced54_cells',weighted_cells,wanted['weighted_cells'])
            originals=npy(folder/'original_objectives.npy');check('original_objectives_schema',originals.shape==(3000,2) and originals.dtype==np.float64 and np.isfinite(originals).all())
            close('nominal_cells',progress[:,0],cell_values['nominal'].mean(axis=1),nominal=True)
            close('balanced_cells',progress[:,1],weighted_cells.mean(axis=1));close('physical_cells',progress[:,2],cell_values['physical'].mean(axis=1))
            close('original_full_cells',originals[:,0],cell_values['full_state'].mean(axis=1))
            close('balanced_total',progress[:,3],(progress[:,0]+WEIGHT*progress[:,1])+progress[:,2])
            close('original_total',originals[:,1],(progress[:,0]+WEIGHT*originals[:,0])+progress[:,2])
            close('nominal_f32_mean',progress[:,0],wanted['nominal'],nominal=True)
''')
old="original1000_weights_exact=True,all_other_weights_exact=True,new323_columns_zero=True,original1000_normalization_exact=True,fresh_optimizer=True,optimizer_step=0,RNG_exact=True,initial_drift="
new="actor_exact=True,learned_context_columns_preserved=True,normalization_exact=True,warm_optimizer_exact=True,optimizer_state_count=6,optimizer_start_step=3000,fresh_optimizer=False,RNG_exact=True,learning_rate_restart_after_restoration=[1e-5,1e-6],initial_drift="
replace(old,new)
replace("corpora=restored,changed_MatMul_dimension=False", "corpora=restored,source_ordinary_step=68000,changed_MatMul_dimension=False")
replace("m=metrics(outputs[label],data)","m=balanced_metrics(metrics(outputs[label],data),group_weights,energy)")
replace("            check(condition+'_initial_drift_gate',maximum<=1e-5)","            check(condition+'_initial_drift_gate',maximum<=1e-5)\n            compare('report_initial_restoration',report['initial_restoration'],restoration['initial_drift'])\n            compare('report_group_weights',report['response_group_weights'],group_weights.tolist())\n            check('report_weight_rule',report['response_weight_rule']=='Emean/Eg')\n            compare('report_direct_subjects',report['direct_subject_sha256'],{key:subject['sha256'] for key,subject in request['subjects'].items()})\n            compare('training_columns',report['training_loss_columns'],['nominal','balanced_full_state','physical','balanced_total','learning_rate','preclip_gradient_norm'])")
block("        stage='paired_comparison'", "        stage='process_accounting'", "        qualified=qualification['causal']\n        final=calculated['causal']['ORT64']\n")
replace("check('known_successful_exit',exit_record['exit_known'] is True and exit_record['child_started'] is True and exit_record['all_postrun_pins_exact'] is True and exit_record['raw_python_exit_code']==exit_record['exit_code']==0 and exit_record['error'] is None)", "check('known_consistent_exit',exit_record['exit_known'] is True and exit_record['child_started'] is True and exit_record['all_postrun_pins_exact'] is True and exit_record['raw_python_exit_code']==exit_record['exit_code']==(0 if qualified else 1) and (not qualified or exit_record['error'] is None))")
replace("        required_pins=dict(frozen['input_sha256']);", "        check('exit_clearance',exit_record['clearance_sha256']==sha(base/'training_clearance.json'))\n        required_pins=dict(frozen['input_sha256']);")
block("        check('no_failed_stage'", "        stage='final_rehash'",'''        if qualified:
            check('no_failed_stage',not (fit/'failure.json').exists() and not (fit/'setup_failure.json').exists())
        else:
            failure=read(bind(fit/'failure.json'))
            check('preserved_numerical_failure',failure['optimization_completed'] is True and failure['final_export_diagnostics_completed'] is True and failure['automatic_retry'] is False and failure['ordinary_endpoint_preserved'] is True and failure['all_frozen_inputs_unchanged'] is True and failure['preservation_errors']==[] and failure['error']=="ValueError('Endpoint same-weight FP64 parity failed; continuation incomplete.')")
        role_paths=release_paths(base,request)
        for role,path in role_paths.items():literal(ar['subjects'][role],path,role)
        owner_path=base/'owner_completion_verification.json';literal(ar['subjects']['owner_completion'],owner_path,'owner_completion');owner=read(owner_path)
        for name in ('owner_verification_passed','accounting_passed','processes_absent','raw_exit_known','all_postrun_pins_exact'):check('owner_flag:'+name,owner[name] is True)
        check('owner_qualification',owner['completion_passed'] is qualified and owner['numerical_completion_passed'] is qualified and owner['optimization_completed'] is True)
        check('owner_exit_and_pids',owner['raw_python_exit_code']==owner['exit_code']==exit_record['exit_code'] and owner['expected_pids']==[exit_record['wrapper_pid'],exit_record['child_pid']])
        current_pins={canonical(p):d for p,d in frozen['input_sha256'].items()};current_pins.update({canonical(Path(frozen['source_directory'])/n):d for n,d in frozen['source_sha256'].items()})
        check('owner_counts',owner['launch_pin_count']==len(expected_pins) and owner['current_pin_count']==len(current_pins) and owner['condition']=='causal')
        owner_expected={role:sha(path) for role,path in role_paths.items()}
        owner_expected.update(process_exit=sha(base/'fit_process_v1/exit.json'),source_fit_report=request['subjects']['fit_report']['sha256'],balance_source_review=request['subjects']['balance_source_review']['sha256'])
        if not qualified:owner_expected['failure']=sha(fit/'failure.json')
        compare('owner_direct_subjects',owner['direct_subject_sha256'],owner_expected)
''')
replace("paired_complete=True,export_qualified=qualified,condition_export_qualified=qualification", "optimization_complete=True,export_qualified=qualified")
replace("paired_report_sha256=sha(fit/'paired_report.json')", "fit_report_sha256=sha(fit/'report.json')")
replace("graph=graph_reports,final_comparison=final", "graph=graph_reports,final_metrics=final")
replace("condition_subjects=condition_subjects,input_sha256=tracked", "input_sha256=tracked")
replace("paired_complete=False,export_qualified=False", "optimization_complete=False,export_qualified=False")
(DEST/'audit_saved_warm.py').write_text(text)
(BASE/'source_delta.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='qualified/audit_saved_pair.py',tofile='source_draft_v1/audit_saved_warm.py')))
tests=(OLD/'test_pair_audit.py').read_text().replace("'audit_saved_pair.py'","'audit_saved_warm.py','audit_balanced_math.py','audit_release.py'")
(DEST/'test_prior_math.py').write_text(tests)
(BASE/'derivation.json').write_text(json.dumps(dict(source=OLD.as_posix(),source_sha256=sha(OLD/'audit_saved_pair.py'),changed_blocks=changes,
    copied_math_sha256={p.name:sha(p) for p in DEST.glob('audit_*.py') if p.name not in ('audit_saved_warm.py','audit_balanced_math.py','audit_release.py')}),indent=2))
print('derived source only')
