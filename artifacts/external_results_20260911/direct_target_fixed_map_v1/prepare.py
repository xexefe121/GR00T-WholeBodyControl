"""Derive saved-only all-controls committed-map diagnostic; no runtime calls."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
prior=NEW/'student_physical_response_fixed_map_v1/diagnose_fixed_map.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
text=prior.read_text();changes=[]
def replace(old,new):
    global text
    assert old in text,old
    changes.append(dict(old=old,new=new,occurrences=text.count(old)))
    text=text.replace(old,new)
replace('All seventy actual learned controls','All actual direct-target learned controls')
replace("NEW/'one_step_physical_student_evaluation_v1/nominal/trace.npz'","NEW/'direct_target_student_evaluation_v1/nominal/trace.npz'")
replace("NEW/'student_physical_response_saved_outcome_v1/report.json'","NEW/'direct_target_saved_outcome_review_v1/report.json'")
replace("NEW/'student_physical_response_independent_physics_v1/report.json'","local(args.root_physics)")
replace("NEW/'one_step_physical_student_evaluation_v1/nominal/report.json'","NEW/'direct_target_student_evaluation_v1/nominal/report.json'")
replace("NEW/'one_step_physical_student_v1/fit/report.json'","NEW/'direct_target_student_v1/fit/report.json'")
replace("    assert sha(p['actual'])=='38428ae27055be7c2e771c5a23056c858f004ef38f260a0b45a57662d8d0aae3'\n    assert read(p['prior_outcome_audit'])['moving_controls']==70\n    assert sha(p['root_physics'])=='ede8ff939da69885f509785e62e6c6cc8d0ecb190b2cb8fb15eb1e5ec4394133'", "    assert args.actual_sha and args.root_physics_sha and args.outcome_sha\n    assert sha(p['actual'])==args.actual_sha\n    assert sha(p['root_physics'])==args.root_physics_sha\n    assert sha(p['prior_outcome_audit'])==args.outcome_sha\n    actual=load(p['actual']);controls=actual['global_control'][actual['controller_mode']==1].astype(int).tolist()\n    assert controls and controls==list(range(250,250+len(controls))) and controls[-1]<1269\n    assert actual['features'].shape==(len(actual['target']),1000)\n    assert read(p['prior_outcome_audit'])['moving_controls']==len(controls)")
replace("paths={k:str(v) for k,v in p.items()},controls=list(range(250,320)),\n        nominal_map_checks=70,actual_state_map_evaluations=70", "paths={k:str(v) for k,v in p.items()},controls=controls,\n        nominal_map_checks=len(controls),actual_state_map_evaluations=len(controls)")
replace('dataset2 centers2038..2107','dataset2 center2038 + control-250')
replace("actor_history_deviation_rms=rms(a['history'][control]-centers['history'][idx])", "terminal_BFM_history_deviation_rms=rms(a['history'][control]-centers['history'][idx])")
replace('failing_left_ankle_roll=dict','left_ankle_roll=dict')
replace('nominal_map_checks_exact=70,actual_state_map_evaluations=70',"nominal_map_checks_exact=len(records),actual_state_map_evaluations=len(records)")
replace('The maps do not consume prior/history, replan, establish local feasibility, or provide expert truth for those off-trajectory states.', 'The maps do not consume prior/history, replan, establish local feasibility, or provide expert truth for those off-trajectory states. The direct learned head uses only1000 measured-state/prepared-goal features; reported prior/history differences are terminal-handoff bookkeeping, not head inputs.')
replace("parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()", "parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')\n    parser.add_argument('--root-physics');parser.add_argument('--root-physics-sha');parser.add_argument('--actual-sha');parser.add_argument('--outcome-sha');args=parser.parse_args()")
ast.parse(text)
out=BASE/'diagnose_fixed_map.py'
with out.open('x',encoding='utf-8') as f:f.write(text)
with (BASE/'source_derivation.json').open('x') as f:json.dump(dict(prior_source=str(prior),prior_sha256=sha(prior),derived_source=str(out),derived_sha256=sha(out),changes=changes,
    invariant='Original AST extraction of quat_mul/quat_log/Planner.difference, full K multiplication and both clip expressions remain literal unchanged.',
    model_calls=0,native_steps=0,optimizer_updates=0),f,indent=2)
print(json.dumps(dict(source_sha256=sha(out),substitutions=len(changes),executed=False)))
