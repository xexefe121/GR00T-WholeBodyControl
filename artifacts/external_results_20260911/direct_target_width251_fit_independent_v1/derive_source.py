"""Source-only derivative; no task arrays, checkpoints, numerical imports or execution."""
from pathlib import Path
import ast,hashlib,json,difflib
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_width512_fit_independent_v1/source_prepared_v1'
DEST=BASE/'source_draft_v1';DEST.mkdir(exist_ok=True)
for p in OLD.glob('*.py'):
    if p.name not in ('audit_saved_warm.py',):
        target=DEST/p.name
        if not target.exists():target.write_bytes(p.read_bytes())
text=(OLD/'audit_saved_warm.py').read_text();original=text
def replace(before,after,count=1):
    global text
    assert text.count(before)==count,(before,text.count(before),count)
    text=text.replace(before,after)
replace('Audit saved width81000 evidence.','Audit saved D3 warm512 ordinary91000 evidence.')
replace('from audit_full_state_math import BACKENDS,CORPORA,SIZES,schedule,numerical_comparisons,drift_summary',
    'from audit_full_state_math import schedule,drift_summary\nfrom audit_recovery_math import BACKENDS,CORPORA,SIZES,OLD_CORPORA,warm_initial_errors,restoration_fields,recovery_schema,recovery_metrics,recovery_ledger_expectations,numerical_comparisons')
replace('from audit_width_math import rate,width_initial_errors,restoration_fields','from audit_width_math import rate')
replace("nominal=name.endswith(('normalized_MSE','nominal_objective','weighted_objective'))", "nominal=name.endswith(('normalized_MSE','nominal_objective','weighted_objective','recovery_objective'))")
replace("'saved_width512_warm_only'","'saved_width251_recovery_warm_only'")
start=text.index('        expected_budget=dict(');end=text.index("        clearance=read(",start)
text=text[:start]+'''        expected_budget=dict(training_forward_rows=157040000,training_forward_calls=40000,training_updates=10000,
            diagnostic_Torch_rows=1474352,diagnostic_Torch_calls=5764,diagnostic_ORT_rows=368588,diagnostic_ORT_calls=1441,
            calibration_forward_calls=0,calibration_gradient_calls=0,native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
        for name,value in dict(kind='qualified_width251_recovery_warm512_fit',root_selected=True,condition='causal',
            updates=10000,ordinary_start_step=81000,ordinary_final_step=91000,optimizer_start_step=16000,optimizer_final_step=26000,
            fresh_optimizer=False,coefficient=WEIGHT,budgets=expected_budget,weight_decay=1e-5,gradient_clip=10.,
            coefficient_recalibration=False,initial_byte_gate_required=False,initial_parity_tolerance_rad=1e-5,parity_tolerance_rad=1e-5,
            features=1323,architecture=[1323,512,512,23],context_order='previous_action23_then_incoming_history300',
            first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',
            automatic_retry=False,no_checkpoint_selection=True,context_and_normalization_reused=True,expansion_performed=False,
            recovery_rows=1018,recovery_phase_counts=[99,819,100],recovery_coefficient=.2,
            recovery_objective='equal_three_phase_normalized_MSE',response_schedule='unchanged10000_prefix_no_wrap',
            consistency_evidence_reviewed=True).items():compare('request.'+name,request[name],value)
        compare('request.learning_rate_values',request['learning_rate_values'],[rate(i) for i in range(10000)])
''' +text[end:]
replace("'schedule_lineage.json','reused_inputs.json','runtime.json']","'schedule_lineage.json','reused_inputs.json','runtime.json','recovery_rows.npz','recovery_evidence.json']")
replace("prior_axes_sha256=sha(request['context_paths']['schedule_axes'])))", "prior_axes_sha256=sha(request['context_paths']['schedule_axes']),selected_prefix_updates=10000,wrap_or_regeneration=False))")
replace("        runtime=read(shared/'runtime.json')",'''        recovery=archive(bind(request['subjects']['recovery_rows']['path'],request['subjects']['recovery_rows']['sha256']))
        rcells=recovery_schema(recovery,limits);check('qualified_recovery_schema',True)
        check('copied_recovery_exact',sha(bind(shared/'recovery_rows.npz'))==request['subjects']['recovery_rows']['sha256'])
        data.update(recovery_teacher=recovery['expert_target'],recovery_cells=rcells)
        validate_recovery_bindings(request,read,bind,check,compare,sha,canonical,frozen['source_sha256'])
        compare('recovery_evidence',read(shared/'recovery_evidence.json'),dict(
            collection_report_sha256=request['subjects']['collection_report']['sha256'],consistency_report_sha256=request['subjects']['consistency_report']['sha256'],
            conflicts_not_automatically_repaired=True,consistency_reviewed_by_selected_fit=True,fresh_student_state_queries=1,connected_expert_rows=1018))
        runtime=read(shared/'runtime.json')''')
replace('from audit_release import release_paths','from audit_release import release_paths\nfrom audit_recovery_bindings import validate_recovery_bindings')
replace('from audit_recovery_bindings import validate_recovery_bindings','from audit_recovery_bindings import validate_recovery_bindings\nfrom audit_fit_process import validate_process_chain')
replace("'source_schedule_centers.npy','source_schedule_axes.npy'", "'source_source_schedule_centers.npy','source_source_schedule_axes.npy'")
replace("npy(shared/('source_schedule_'+key+'.npy'))", "npy(shared/('source_'+Path(request['context_paths']['schedule_'+key]).name))")
replace("        old_frozen=read(bind(request['subjects']['source_frozen_inputs']['path']))\n        compare('context_source_map',alignment['source_input_sha256'],old_frozen['input_sha256'])\n", '')
replace("check('source71000',source['ordinary_final_step']==71000 and request['subjects']['checkpoint']['sha256']=='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d')", "check('source81000',source['ordinary_final_step']==81000 and request['subjects']['checkpoint']['sha256']=='825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e')")
replace("        source_report=read(bind(request['subjects']['fit_report']['path']))", "        original_context_receipt=source['request']['subjects']['source_frozen_inputs']\n        check('original_context_receipt_frozen',frozen_members.get(canonical(original_context_receipt['path']))==original_context_receipt['sha256'])\n        old_frozen=read(bind(original_context_receipt['path'],original_context_receipt['sha256']))\n        compare('context_source_map',alignment['source_input_sha256'],old_frozen['input_sha256'])\n        source_report=read(bind(request['subjects']['fit_report']['path']))")
replace("source_report['ordinary_final_step']==71000", "source_report['ordinary_final_step']==81000")
replace("group_weights=energy_weights(energy)","group_weights=energy_weights(energy)\n        exact('request_group_weights',np.asarray(request['group_weights'],np.float64),group_weights)\n        exact('source_group_weights',np.asarray(source['response_group_weights'],np.float64),group_weights)")
replace("'promoted_parameters.json']","'promoted_parameters.json','recovery_objectives.npy','recovery_cell_losses.npy']")
replace('ordinary_final_step=81000,additional_updates=10000,optimizer_step=16000','ordinary_final_step=91000,additional_updates=10000,optimizer_step=26000',2)
replace("ordinary_start_step=71000,optimizer_start_step=6000,hidden_width=512,architecture=[1323,512,512,23],expansion_seed=20260912", "ordinary_start_step=81000,optimizer_start_step=16000,hidden_width=512,architecture=[1323,512,512,23],expansion_performed=False,recovery_rows=1018,recovery_phase_counts=[99,819,100],recovery_coefficient=.2")
replace("check('width_initial_actor_optimizer_RNG',not width_initial_errors(init,source))", "check('warm_full512_initial_actor_optimizer_RNG',not warm_initial_errors(init,source))")
replace("tree('retained_expansion_generator',checkpoint['expansion_generator_state'],init['expansion_generator_state'])", "tree('source_expansion_provenance',checkpoint['source_expansion_generator_state'],source['expansion_generator_state'])\n            compare('source_expansion_seed',checkpoint['source_expansion_seed'],source['expansion_seed'])")
replace('ordinary_start_step=71000,optimizer_start_step=6000,hidden_width=512,expansion_seed=20260912', 'ordinary_start_step=81000,optimizer_start_step=16000,hidden_width=512,expansion_performed=False')
replace('architecture=[1323,512,512,23],expansion_seed=20260912,context_order=', 'architecture=[1323,512,512,23],expansion_performed=False,recovery_coefficient=.2,context_order=')
replace("float(state['step'])==16000", "float(state['step'])==26000")
replace("optimization['ordinary_final_step']==81000 and optimization['optimizer_step']==16000", "optimization['ordinary_final_step']==91000 and optimization['optimizer_step']==26000")
replace('for name,width in zip(CORPORA,(15,54,9)):', 'for name,width in zip(OLD_CORPORA,(15,54,9)):')
replace("            outputs={label:", '''            dc=npy(folder/'recovery_cell_losses.npy');dl=npy(folder/'recovery_objectives.npy')
            check('D3_ledger_shapes',dc.shape==dl.shape==(10000,3))
            de=recovery_ledger_expectations(dc,dl,progress[:,3],.2)
            close('D3_equal_phase_mean',dl[:,0],de['phase_mean'],nominal=True)
            exact('D3_weighted_float32',dl[:,1],de['weighted']);close('D3_combined_total',dl[:,2],de['combined'])
            compare('D3_columns',report['recovery_loss_columns'],['recovery','weighted_recovery','combined_total'])
            compare('D3_collection',report['recovery_collection'],request['subjects']['collection_report'])
            outputs={label:''')
replace("for corpus in CORPORA}\n            maximum=max(v['max_preclip_error_rad']", "for corpus in OLD_CORPORA}\n            maximum=max(v['max_preclip_error_rad']")
replace('source_ordinary_step=71000,changed_MatMul_dimension=True', 'source_ordinary_step=81000,changed_MatMul_dimension=False')
replace('m=balanced_metrics(metrics(outputs[label],data),group_weights,energy)', 'm=recovery_metrics(outputs[label],data,group_weights,energy,.2)')
replace("condition+'_nine_parity'", "condition+'_twelve_parity'")
replace("condition+'_nine_drifts',len(drift_report)==9", "condition+'_twelve_drifts',len(drift_report)==12")
replace('counts(30000,146860000)','counts(40000,157040000)')
replace('counts(1437,367570)','counts(1441,368588)',2)
replace("condition+'_7185_call_partitions',ledger==expected and len(ledger)==7185", "condition+'_7205_call_partitions',ledger==expected and len(ledger)==7205")
replace("        expected_pins={canonical(p):d for p,d in required_pins.items()}", "        launch_path=bind(base/'launch_receipt.json');launch_sha=sha(launch_path);launch=read(launch_path)\n        check('clearance_launch_receipt',clearance['launch_receipt_sha256']==launch_sha and launch_review['launch_receipt_sha256']==launch_sha)\n        for key,value in dict(request_sha256=training_sha,frozen_receipt_sha256=frozen_sha,launcher_sha256=clearance['launcher_sha256'],automatic_retry=False).items():compare('launch.'+key,launch[key],value)\n        static_pins={canonical(p):d for p,d in required_pins.items() if canonical(p) not in {canonical(base/'training_clearance.json'),canonical(clearance['review_path'])}}\n        compare('static_launch_pins',{canonical(p):d for p,d in launch['input_sha256'].items()},static_pins)\n        check('static_launch_count',launch['pin_count']==len(static_pins)==len(launch['input_sha256']))\n        required_pins[str(launch_path)]=launch_sha\n        expected_pins={canonical(p):d for p,d in required_pins.items()}")
replace("        compare('owner_direct_subjects',owner['direct_subject_sha256'],owner_expected)", "        compare('owner_direct_subjects',owner['direct_subject_sha256'],owner_expected)\n        validate_process_chain(base,request,clearance,launch_sha,start,child,exit_record,owner,owner_expected,read,bind,sha,check,compare,canonical)")
replace('Initialization checks reconstruct saved tensors with a local CPU generator; they do not execute a model. New moment entries retain shared age6000.',
    'Initialization requires every saved full512 actor/moment/group/RNG tensor exact at age16000; no model or expansion is executed.')
replace('Initial dimensional drift retains the original1e-5rad gate.','Original three-corpus initial drift retains the1e-5rad gate; recovery has no prior prediction byte gate.')
replace('Nominal float32 reductions use3e-7 relative tolerance;', 'Nominal and D3 float32 reductions use3e-7 relative tolerance;')
ast.parse(text);(DEST/'audit_saved_warm.py').write_text(text,encoding='utf-8',newline='\n')
diff=''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='width81000/audit_saved_warm.py',tofile='recovery91000/audit_saved_warm.py'))
(BASE/'main_derivation.patch').write_text(diff,encoding='utf-8',newline='\n')
print(json.dumps(dict(main_sha256=hashlib.sha256(text.encode()).hexdigest(),source_only=True,task_calls=0)))
