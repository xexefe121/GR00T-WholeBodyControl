"""Draft source derivation only. Never constructs a model or training request."""
from pathlib import Path
import hashlib,json,shutil

BASE=Path(__file__).resolve().parent
PRIOR=BASE.parent/'velocity_chord_student_v1'
OLD=PRIOR/'source_snapshot_v3'
DEST=BASE/'source_draft_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original={}
for p in sorted(OLD.rglob('*.py')):
    relative=p.relative_to(OLD);q=DEST/relative;q.parent.mkdir(parents=True,exist_ok=True)
    assert not q.exists() or q.read_bytes()==p.read_bytes(),str(relative)
    if not q.exists():shutil.copyfile(p,q)
    original[relative.as_posix()]=sha(p)
source=(OLD/'fit_velocity_chords.py').read_text(encoding='utf-8')
changes=[]
def replace(old,new,count=1):
    global source
    assert source.count(old)==count,(old[:100],source.count(old),count)
    source=source.replace(old,new)
    changes.append(dict(old=old,new=new,count=count))

replace('One fixed ordinary65000→70000 continuation, gated on reviewed generated data.',
        'Draft ordinary70000→75000 continuation; requires real final reviewed branch inputs.')
replace('from chord_fit_diagnostics import evaluate_fixed,complete_chord_metrics,nominal_metrics',
'''from chord_fit_diagnostics import complete_chord_metrics,nominal_metrics
from physical_fit_diagnostics import evaluate_all,physical_metrics
from physical_training_inputs import load_physical,validate_bound_review
from physical_training_objective import physical_response_loss,expected_budgets''')
replace("PRIOR=NEW/'fast_controller_phase_fit_v1'", "PRIOR=NEW/'velocity_chord_student_v1'\nLEGACY=NEW/'fast_controller_phase_fit_v1'\nGENERATION=PRIOR/'generation'")
replace("completed_steps=65000", "completed_steps=70000")
replace("ACTIVE={}\n", """ACTIVE={}
STATE['legacy_diagnostics']=dict(head_onnx_calls_attempted=0,head_onnx_calls_returned=0,
    diagnostic_torch_rows_attempted=0,diagnostic_torch_rows_returned=0)
STATE['physical_diagnostics']=dict(head_onnx_calls_attempted=0,head_onnx_calls_returned=0,
    diagnostic_torch_rows_attempted=0,diagnostic_torch_rows_returned=0)
""")
start=source.index('def check_inputs():');end=source.index('\ndef main():',start)
old=source[start:end]
new='''def check_inputs():
    # No model/optimizer/graph is constructed until every actual launch gate passes.
    receipt=read(BASE/'training_frozen_inputs.json')
    request=read(BASE/'training_request.json')
    assert sha(BASE/'training_request.json')==receipt['training_request_sha256']
    assert request['kind']=='one_physical_response_continuation_from70000'
    assert request['additional_updates']==5000 and request['ordinary_final_step']==75000
    assert request['coefficients']==dict(nominal=1.,velocity=1.,physical=1.)
    assert request['budgets']==expected_budgets(request['valid_rows'])
    assert 0<request['valid_rows']<=3054
    assert request['input_sha256']==receipt['input_sha256']
    for name,digest in receipt['source_sha256'].items():assert sha(Path(__file__).parent/name)==digest,name
    for name,digest in receipt['input_sha256'].items():assert sha(local(name))==digest,name
    clearance=read(BASE/'training_clearance.json')
    assert clearance['approved'] is True and clearance['model_fitting_authorized'] is True
    assert clearance['additional_updates']==5000 and clearance['ordinary_final_step']==75000
    assert clearance['frozen_receipt_sha256']==sha(BASE/'training_frozen_inputs.json')
    assert clearance['training_request_sha256']==sha(BASE/'training_request.json')
    assert sha(local(clearance['source_review_path']))==clearance['source_review_sha256']
    assert clearance['source_review_sha256']==request['reviews']['source']['sha256']
    assert set(request['reviews'])=={'prior_fit','prior_export','branch_data','source'}
    for spec in request['reviews'].values():validate_bound_review(spec,receipt['input_sha256'])
    assert request['reviews']['prior_fit']['subjects'][str(PRIOR/'fit/student_head.pt').replace('\\\\','/')]==sha(PRIOR/'fit/student_head.pt')
    assert request['reviews']['prior_export']['subjects'][str(PRIOR/'fit/student_head.onnx').replace('\\\\','/')]==sha(PRIOR/'fit/student_head.onnx')
    assert request['reviews']['branch_data']['subjects'][request['physical_paths']['manifest']]==sha(local(request['physical_paths']['manifest']))
    report=read(PRIOR/'fit/report.json')
    assert report['ordinary_final_step']==70000 and report['numerical_gate_passed'] is True
    assert report['checkpoint_sha256']==sha(PRIOR/'fit/student_head.pt')
    assert report['onnx_sha256']==sha(PRIOR/'fit/student_head.onnx')
    assert sha(PRIOR/'fit/student_head.pt')=='17b8e240e3ca711e993a5c84c0221326d365f7ed892c8166bb594d804726bc3e'
    assert sha(PRIOR/'fit/student_head.onnx')=='219b86cc4decd51671cb7aa036a944741cebed684accd91f7a03de37a7694bb5'
    generated=read(GENERATION/'report.json')
    assert generated['complete'] is True and generated['all_center_byte_parity'] is True
    assert generated['inference_calls']==dict(actor=143679,backward=3057)
    assert runtime_identity()==read(PRIOR/'training_runtime_identity.json')
    return receipt,clearance,request
'''
replace(old,new)
replace('receipt,clearance=check_inputs()', 'receipt,clearance,training_request=check_inputs()')
replace("centers=archive(BASE/'generation/centers.npz')", "centers=archive(GENERATION/'centers.npz')\n    physical=load_physical(training_request,centers)\n    budget=physical['budgets']")
for name in ('features.npy','base_target.npy','teacher_target.npy'):
    replace(f"BASE/'generation/{name}'",f"GENERATION/'{name}'")
replace("STATE['stage']='RESTORING_65000'", "STATE['stage']='RESTORING_70000'")
replace("saved['completed_steps']==65000", "saved['completed_steps']==70000")
replace('int(s[\'step\'])==65000', 'int(s[\'step\'])==70000')
replace("group['lr']==3e-6", "group['lr']==3e-7")
replace("previous_fit=archive(PRIOR/'fit/teacher_fit.npz')", "previous_fit=archive(LEGACY/'fit/teacher_fit.npz')\n    previous_prediction=archive(PRIOR/'fit/final_predictions.npz')")
replace("DEST.mkdir(exist_ok=False)", """physical_features=torch.from_numpy((physical['features']-mean)/std)
    physical_base=torch.from_numpy(physical['base']);physical_target=torch.from_numpy(physical['target'])
    physical_successors=torch.from_numpy(physical['successor'])
    assert torch.isfinite(physical_features).all()
    DEST.mkdir(exist_ok=False)
    np.savez_compressed(DEST/'physical_row_selection.npz',requested_valid_mask=physical['valid_mask'],
        selected_rows=physical['rows'],successor_center=physical['successor'])
    write(DEST/'physical_coverage.json',physical['coverage'])""")
replace("'restored65000.onnx'", "'restored70000.onnx'",3)
replace("actual_trace=archive(PRIOR/'nominal/trace.npz')", "actual_trace=archive(LEGACY/'nominal/trace.npz')")
replace("before=evaluate_fixed(actor,centers['features'],px,actual,mean,std,span,DEST/'restored70000.onnx',STATE,DEST,'initial')",
"before=evaluate_all(actor,centers['features'],px,actual,mean,std,span,DEST/'restored70000.onnx',STATE,DEST,'initial',physical)")
replace("previous_fit['predicted_delta']", "previous_prediction['predicted_delta'][:3057]")
replace("read(PRIOR/'fit/report.json')['final_nine_cell_objective']", "read(PRIOR/'fit/report.json')['final_nominal_objective']")
replace("pairs,65000)", "pairs,70000)")
replace("write(DEST/'initial_chord_metrics.json',initial_chords)", """write(DEST/'initial_chord_metrics.json',initial_chords)
    assert initial_chords['full_nine_cell_chord_objective']==read(PRIOR/'fit/report.json')['final_chord_objective']
    exact(before['predicted_delta'],previous_prediction['predicted_delta'],'all restored nominal and velocity predictions')
    exact(before['actual_seven_delta'],previous_prediction['actual_seven_delta'],'old seven actual predictions')
    initial_physical=physical_metrics(before['predicted_delta'][:3057],before['physical']['predicted_delta'],centers,physical,span)
    write(DEST/'initial_physical_metrics.json',initial_physical)""")
replace("'restoration65000_parity.json'", "'restoration70000_parity.json'")
replace('request=dict(first_step=65001,last_step=70000,updates=5000,lambda_chord=1.,full_nominal_rows=3057,',
'''request=dict(first_step=70001,last_step=75000,updates=5000,lambda_chord=1.,lambda_physical=1.,full_nominal_rows=3057,
        valid_physical_rows=len(physical['rows']),requested_physical_rows=3054,physical_coverage=physical['coverage'],
        physical_pairing='same dataset nominal successor c+1; live center; unclipped total-target response',
        physical_requested_denominators=[99,819,100]*3,training_request_sha256=sha(BASE/'training_request.json'),
        input_request=training_request,budgets=budget,learning_rate_reset_after_exact_restoration=True,''')
replace("runtime=read(BASE/'training_runtime_identity.json'),runtime_identity_sha256=sha(BASE/'training_runtime_identity.json'),",
"runtime=read(PRIOR/'training_runtime_identity.json'),runtime_identity_sha256=sha(PRIOR/'training_runtime_identity.json'),")
replace('training_head_rows=21045000', "training_head_rows=budget['training_head_rows']")
replace('losses=[];completed=65000','losses=[];completed=70000')
replace('range(65001,70001)', 'range(70001,75001)',2)
replace('loss=nominal+chords', '''STATE['training_head_rows_attempted']+=len(physical['rows'])
            normalized_physical=actor(physical_features);STATE['training_head_rows_returned']+=len(physical['rows'])
            ACTIVE['physical_normalized_returned']=normalized_physical.detach().numpy().copy()
            physical_loss,physical_cell_losses=physical_response_loss(normalized_centers,normalized_physical,
                physical_successors,base_t,physical_base,target_t,physical_target,span_t,physical['cells'])
            ACTIVE['physical_cell_losses']=physical_cell_losses.detach().numpy().copy()
            loss=nominal+chords+physical_loss''')
replace('losses.append([float(nominal.detach()),float(chords.detach()),float(loss.detach()),rate])',
'''losses.append([float(nominal.detach()),float(chords.detach()),float(physical_loss.detach()),float(loss.detach()),rate])''')
replace('combined_objective=losses[-1][2],learning_rate=rate)', 'physical_objective=losses[-1][2],combined_objective=losses[-1][3],learning_rate=rate)')
replace('reshape(-1,4)', 'reshape(-1,5)')
replace('assert completed==70000', 'assert completed==75000')
replace("int(s['step'])==70000", "int(s['step'])==75000",1) if False else None
# Only the final assertion changes; restoration remains step70000.
replace("assert all(int(s['step'])==70000 for s in optimizer.state.values())\n    assert STATE['training_head_rows_attempted']", "assert all(int(s['step'])==75000 for s in optimizer.state.values())\n    assert STATE['training_head_rows_attempted']")
replace("==21045000", "==budget['training_head_rows']")
replace('completed_steps=70000,additional_updates=5000,request=request)', 'completed_steps=75000,additional_updates=5000,request=request)')
replace("training_objective=np.asarray(losses)[:,2],learning_rate=np.asarray(losses)[:,3])",
"physical_objective=np.asarray(losses)[:,2],training_objective=np.asarray(losses)[:,3],learning_rate=np.asarray(losses)[:,4])")
replace('ordinary_final_step=70000', 'ordinary_final_step=75000',2)
replace("final=evaluate_fixed(actor,centers['features'],px,actual,mean,std,span,DEST/'student_head.onnx',STATE,DEST,'final')",
"final=evaluate_all(actor,centers['features'],px,actual,mean,std,span,DEST/'student_head.onnx',STATE,DEST,'final',physical)")
replace('pairs,70000)', 'pairs,75000)',1) if False else None
replace("final_nominal=nominal_metrics(final['normalized_centers'],final['predicted_delta'][:3057],centers,strata,span,pairs,70000)",
"final_nominal=nominal_metrics(final['normalized_centers'],final['predicted_delta'][:3057],centers,strata,span,pairs,75000)")
replace("write(DEST/'final_nominal_metrics.json',final_nominal)", """write(DEST/'final_nominal_metrics.json',final_nominal)
    final_physical=physical_metrics(final['predicted_delta'][:3057],final['physical']['predicted_delta'],centers,physical,span)
    write(DEST/'final_physical_metrics.json',final_physical)""")
replace("checkpoint_sha256=sha(DEST/'student_head.pt'),onnx_sha256=sha(DEST/'student_head.onnx'),",
"""initial_physical_objective=initial_physical['nine_cell_response_objective'],
        final_physical_objective=final_physical['nine_cell_response_objective'],
        physical_coverage=physical['coverage'],physical_data_manifest_sha256=sha(local(training_request['physical_paths']['manifest'])),
        checkpoint_sha256=sha(DEST/'student_head.pt'),onnx_sha256=sha(DEST/'student_head.onnx'),""")
replace("report['head_ONNX_calls']==1126", "report['head_ONNX_calls']==budget['head_onnx_calls']")
replace("==287372", "==budget['diagnostic_torch_rows']")
replace("STATE['stage']='FINAL_FROZEN_INPUT_REHASH'", """assert STATE['legacy_diagnostics']['head_onnx_calls_attempted']==STATE['legacy_diagnostics']['head_onnx_calls_returned']==1126
    assert STATE['head_onnx_calls_attempted']==STATE['head_onnx_calls_returned']==budget['head_onnx_calls']
    STATE['stage']='FINAL_FROZEN_INPUT_REHASH'""")
replace('receipt,clearance,training_request=check_inputs()', '''receipt,clearance,training_request=check_inputs()
    initial_gate_hashes={name:sha(BASE/name) for name in ('training_request.json','training_frozen_inputs.json','training_clearance.json')}''')
replace("    check_inputs()\n    report['attempt_counters']", "    assert all(sha(BASE/name)==digest for name,digest in initial_gate_hashes.items())\n    check_inputs()\n    report['attempt_counters']")
replace('losses=[];completed=70000', '''losses=[];cell_losses=[];completed=70000
    ACTIVE.update(nominal_raw_features=centers['features'],physical_raw_features=physical['features'],
        physical_valid_rows=physical['rows'],physical_successor_rows=physical['successor'])''')
replace("STATE['attempted_step']=step", """STATE['attempted_step']=step
            for key in ('nominal_normalized_returned','velocity_normalized_returned','physical_normalized_returned','physical_cell_losses'):
                ACTIVE.pop(key,None)
            STATE['training_forward_stage']='nominal'""")
replace("selected_x=px[ids,axis].reshape(1152,1069)", "selected_x=px[ids,axis].reshape(1152,1069)\n            ACTIVE['velocity_raw_features']=selected_x")
replace("normalized_centers=actor(features);STATE['training_head_rows_returned']+=3057", """normalized_centers=actor(features);STATE['training_head_rows_returned']+=3057
            ACTIVE['nominal_normalized_returned']=normalized_centers.detach().numpy().copy()
            STATE['training_forward_stage']='velocity'""")
replace("normalized_probes=actor(torch.from_numpy((selected_x-mean)/std));STATE['training_head_rows_returned']+=1152", """normalized_probes=actor(torch.from_numpy((selected_x-mean)/std));STATE['training_head_rows_returned']+=1152
            ACTIVE['velocity_normalized_returned']=normalized_probes.detach().numpy().copy()""")
replace("STATE['training_head_rows_attempted']+=len(physical['rows'])", "STATE['training_forward_stage']='physical'\n            STATE['training_head_rows_attempted']+=len(physical['rows'])")
replace("loss=nominal+chords+physical_loss", """loss=nominal+chords+physical_loss
            # Reporting from returned tensors only; original loss graph remains unchanged.
            with torch.no_grad():
                nominal_square=(normalized_centers-labels)**2
                nominal_cells=torch.stack([nominal_square[ix].mean() for group in strata for ix in group])
                center_proposal=base_t[picked]+normalized_centers[picked]*span_t
                velocity_proposal=torch.from_numpy(np.asarray(pb[ids,axis]))+(normalized_probes*span_t).reshape(576,2,23)
                velocity_change=torch.from_numpy(np.asarray(pt[ids,axis]))-target_t[picked,None]
                velocity_square=(((velocity_proposal-center_proposal[:,None])-velocity_change)/span_t)**2
                velocity_cells=torch.stack([velocity_square[cell*64:(cell+1)*64].mean() for cell in range(9)])
                current_cells=np.stack([nominal_cells.numpy(),velocity_cells.numpy(),physical_cell_losses.detach().numpy()])
            ACTIVE['current_cell_losses']=current_cells.copy()""")
replace("losses.append([float(nominal.detach()),float(chords.detach()),float(physical_loss.detach()),float(loss.detach()),rate])", """losses.append([float(nominal.detach()),float(chords.detach()),float(physical_loss.detach()),float(loss.detach()),rate])
            cell_losses.append(current_cells.copy())""")
replace("np.save(DEST/'training_progress.npy',np.asarray(losses))", "np.save(DEST/'training_progress.npy',np.asarray(losses))\n                np.save(DEST/'training_cell_losses.npy',np.asarray(cell_losses))")
replace("np.save(DEST/'training_progress.npy',np.asarray(losses,dtype=np.float64).reshape(-1,5))", "np.save(DEST/'training_progress.npy',np.asarray(losses,dtype=np.float64).reshape(-1,5))\n        np.save(DEST/'training_cell_losses.npy',np.asarray(cell_losses,dtype=np.float64).reshape(-1,3,9))")
replace("nominal_objective=np.asarray(losses)[:,0],sampled_chord_objective=np.asarray(losses)[:,1],", "nominal_objective=np.asarray(losses)[:,0],sampled_chord_objective=np.asarray(losses)[:,1],\n        cell_losses=np.asarray(cell_losses),")
replace("write(DEST/'initial_physical_metrics.json',initial_physical)", """initial_physical['restored_Windows_ORT_vs_collector_WSL_batch1_max_delta']=float(np.max(np.abs(
        before['physical']['onnx_predicted_delta']-physical['collector_head_delta'])))
    initial_physical['cross_platform_comparison_is_not_byte_parity']=True
    write(DEST/'initial_physical_metrics.json',initial_physical)""")
target=DEST/'fit_physical_continuation.py'
if target.exists():
    previous=json.loads((BASE/'source_derivation.json').read_text(encoding='utf-8'))
    assert sha(target)==previous['derived_trainer_sha256']
target.write_text(source,encoding='utf-8')
report=dict(preparation_only=True,optimizer_updates=0,model_evaluations=0,physics_steps=0,
    original_sources_sha256=original,original_trainer_sha256=sha(OLD/'fit_velocity_chords.py'),
    derived_trainer_sha256=sha(target),exact_substitutions=changes,
    no_training_request_or_clearance_created=True)
(BASE/'source_derivation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(originals=len(original),substitutions=len(changes),derived_sha256=sha(target))))
