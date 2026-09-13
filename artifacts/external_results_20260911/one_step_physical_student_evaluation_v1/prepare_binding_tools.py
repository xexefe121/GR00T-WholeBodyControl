"""Derive binding/launcher preparation utilities; never bind or launch a model."""
import ast,json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'velocity_chord_student_evaluation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
changes=[]
def replace(text,old,new,label):
    assert text.count(old)==1,(label,old[:90]);changes.append(dict(file=label,old=old,new=new));return text.replace(old,new)
name='freeze_final_package.py';text=(OLD/name).read_text()
text=text.replace('70000','75000').replace('evaluate_velocity_chord_student.py','evaluate_physical_response_student.py')
text=replace(text,"FIT=BASE.parent/'velocity_chord_student_v1'","FIT=BASE.parent/'one_step_physical_student_v1'\nCENTERS=BASE.parent/'velocity_chord_student_v1/generation/centers.npz'",name)
text=replace(text,"    assert config['root_selected'] is True", "    assert config['root_selected'] is True\n    assert config['ordinary_final_step']==75000 and config['stage']==args.mode\n    assert config['requested_main_controls']==(1569 if args.mode=='evaluation' else 0)\n    assert config['conditional_hold_controls']==(250 if args.mode=='evaluation' else 0)\n    assert config['separate_head_calls']==(1 if args.mode=='witness' else 0)\n    assert config['hardware_authorized'] is False",name)
text=replace(text,"        centers=FIT/'generation/centers.npz',query250_labels=BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz').items()}","        centers=CENTERS,query250_labels=BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz').items()}\n    training_request=read(FIT/'training_request.json')\n    for key in ('manifest','report','collection_request'):\n        final['physical_'+key]=item(training_request['physical_paths'][key])\n    final['training_request']=item(FIT/'training_request.json')\n    for key in ('head','checkpoint','fit_report','training_manifest'):\n        assert config['selected_subject_sha256'][key]==final[key]['sha256'],key",name)
text=replace(text,"        reviews[role]=entry", "        if role=='dataset':\n            for subject in ('centers','physical_manifest','physical_report','physical_collection_request'):\n                assert has_hash(record,final[subject]['sha256']),subject\n        reviews[role]=entry",name)
text=replace(text,"    for path in (Path(__file__),BASE/'prepare_bound_launcher.py',BASE/'original_sources.json',BASE/'derive_evaluator.py',\n                 BASE/'evaluator_derivation.json',BASE/'test_evaluation_package.py',BASE/'focused_final.xml',args.reviews):", "    for path in (Path(__file__),BASE/'prepare_bound_launcher.py',BASE/'prepare_binding_tools.py',BASE/'binding_tools_derivation.json',\n                 BASE/'source_preparation.json',BASE/'source_only_check.json',BASE/'source_freeze.json',BASE/'freeze_sources.py',\n                 BASE/'test_binding_workflow.py',BASE/'binding_workflow_tests.json',BASE/'binding_workflow_tests.log',args.reviews):",name)
text=replace(text,"        onnx_dependencies=DEPS.as_posix(),training_dataset_sha256=[CENTERS_SHA],", "        onnx_dependencies=DEPS.as_posix(),training_dataset_sha256=[CENTERS_SHA,final['physical_manifest']['sha256'],final['physical_report']['sha256']],",name)
(BASE/name).write_text(text);ast.parse(text)
name='prepare_bound_launcher.py';text=(OLD/name).read_text().replace('70000','75000').replace('evaluate_velocity_chord_student.py','evaluate_physical_response_student.py')
text=replace(text,"    launcher=folder/'run.ps1';durable=folder/'run_durable.ps1'", "    launcher=folder/'run.ps1';durable=folder/'run_durable.ps1'\n    # Duplicate-output guards apply before starting the actual WSL command.\n    forbidden=['head_witness'] if mode=='witness' else ['nominal','post_lifecycle_hold_5s','pilot_outcome.json']\n    no_outputs=''.join(\"if (Test-Path -LiteralPath \"+quote(BASE/name)+\") { throw 'Existing attempt output must be preserved.' }\\n\" for name in forbidden)",name)
text=replace(text,"    write(launcher,guard+command+verdict+'exit 0\\n')", "    write(launcher,guard+no_outputs+command+verdict+'exit 0\\n')",name)
text=replace(text,"    guard=HASH+", "    no_outputs+=\"$executionLock = [System.IO.File]::Open(\"+quote(folder/'execution.lock')+\",[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)\\n$executionLock.Close()\\n\"\n    guard=HASH+",name)
# Preserve the original handle-acquired exit-code logic and completed/incomplete verdicts.
(BASE/name).write_text(text);ast.parse(text)
(BASE/'binding_tools_derivation.json').write_text(json.dumps(dict(preparation_only=True,root_stage_selection_required=True,
    old_sources={p:sha(OLD/p) for p in ('freeze_final_package.py','prepare_bound_launcher.py')},
    new_sources={p:sha(BASE/p) for p in ('freeze_final_package.py','prepare_bound_launcher.py')},changes=changes,
    global_subject_substitutions=['ordinary70000 toordinary75000','evaluate_velocity_chord_student.py toevaluate_physical_response_student.py'],
    models_constructed=0,graph_calls=0,native_steps=0,launched=False),indent=2)+'\n')
print(json.dumps(dict(prepared=True,tools=['freeze_final_package.py','prepare_bound_launcher.py'],model_calls=0,native_steps=0)))
