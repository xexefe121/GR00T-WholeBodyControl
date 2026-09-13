import ast,hashlib,json
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');O=Path(__file__).parent
old=N/'student_velocity_chord_fixed_map_v1/diagnose_fixed_map.py';source=old.read_text();changes=[]
def change(a,b):
    global source
    assert source.count(a)==1,(a,source.count(a));source=source.replace(a,b);changes.append(dict(old=a,new=b))
change('Twelve actual states against matching-clock saved committed maps; pure arrays.', 'All seventy actual learned controls against matching-clock saved committed maps; pure arrays.')
change("actual=NEW/'velocity_chord_student_evaluation_v1/nominal/trace.npz'", "actual=NEW/'one_step_physical_student_evaluation_v1/nominal/trace.npz'")
change("prior_outcome_audit=NEW/'student_velocity_chord_saved_outcome_v1/report.json',", "prior_outcome_audit=NEW/'student_physical_response_saved_outcome_v1/report.json',\n    root_physics=NEW/'student_physical_response_independent_physics_v1/report.json',\n    actual_report=NEW/'one_step_physical_student_evaluation_v1/nominal/report.json',\n    final_fit_report=NEW/'one_step_physical_student_v1/fit/report.json',")
change("assert sha(p['actual'])=='1e8e44a6c405a86138558ea89681fb225b73b2be969e2e7a608903ebb5f80872'", "assert sha(p['actual'])=='38428ae27055be7c2e771c5a23056c858f004ef38f260a0b45a57662d8d0aae3'\n    assert read(p['prior_outcome_audit'])['moving_controls']==70\n    assert sha(p['root_physics'])=='ede8ff939da69885f509785e62e6c6cc8d0ecb190b2cb8fb15eb1e5ec4394133'")
change("controls=list(range(250,262)),", "controls=list(range(250,320)),")
change("nominal_map_checks=12,actual_state_map_evaluations=12", "nominal_map_checks=70,actual_state_map_evaluations=70")
change('dataset2 centers2038..2049', 'dataset2 centers2038..2107')
change("head_target=float(a['target'][control,3]),fixed_map_target=float(target[3]),same_clock_nominal_target=float(nominal_target[3]),feedback_raw=float(raw[3]),feedback=float(feedback[3])))", "head_target=float(a['target'][control,3]),fixed_map_target=float(target[3]),same_clock_nominal_target=float(nominal_target[3]),feedback_raw=float(raw[3]),feedback=float(feedback[3])),\n            failing_left_ankle_roll=dict(head_target=float(a['target'][control,5]),fixed_map_target=float(target[5]),same_clock_nominal_target=float(nominal_target[5]),feedback_raw=float(raw[5]),feedback=float(feedback[5]),actual_precontrol_q=float(a['qpos'][control,12]),actual_precontrol_v=float(a['qvel'][control,11]),nominal_q=float(centers['qpos'][idx,12]),nominal_v=float(centers['qvel'][idx,11])))")
change("nominal_map_checks_exact=12,actual_state_map_evaluations=12", "nominal_map_checks_exact=70,actual_state_map_evaluations=70")
change("write(BASE/'report.json',report);print(json.dumps(records[:4],indent=2))", "assert sha(__file__)==request['source_sha256']\n    for p,h in request['input_sha256'].items():assert sha(local(p))==h\n    write(BASE/'report.json',report);print(json.dumps(records[:4],indent=2))")
change("    else:main()", "    else:\n        try:main()\n        except BaseException as error:\n            write(BASE/'failure.json',dict(error=repr(error),model_inference_calls=0,physics_steps=0,optimizer_updates=0,new_queries=0));raise")
ast.parse(source);out=O/'diagnose_fixed_map.py'
with out.open('x',encoding='utf-8') as f:f.write(source)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with (O/'source_derivation.json').open('x',encoding='utf-8') as f:
    json.dump(dict(original_path=old.as_posix(),original_sha256=sha(old),derived_sha256=sha(out),changes=changes,
        full_tangent_K_matmul_and_two_clips_unchanged=True,controls=70,model_inference_calls=0,physics_steps=0),f,indent=2);f.write('\n')
print(json.dumps(dict(source_sha256=sha(out),changes=len(changes))))
