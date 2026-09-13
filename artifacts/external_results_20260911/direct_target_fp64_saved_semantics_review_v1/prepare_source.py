"""Derive the actual42-row pure semantic audit; do not execute it here."""
import ast,difflib,hashlib,json,re
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');B=Path(__file__).parent
oldpath=N/'direct_target_saved_outcome_review_v1/audit_saved.py';old=oldpath.read_text();digest=hashlib.sha256(oldpath.read_bytes()).hexdigest()
assert digest=='e707ee4c3ffd828e9347dbfde1ddb5937cb3507b03bdbec57d26b52d8dda609f'
text=old;changes=[]
def replace(a,b):
 global text
 assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b);changes.append({'before':a,'after':b})
replace("run=NEW/'direct_target_student_evaluation_v1';src=run/'source_draft_v1'","run=NEW/'direct_target_fp64_export_evaluation_v2';src=run/'source_draft_v1'")
replace("norm=NEW/'direct_target_student_v1/fit/normalization.npz'","norm=NEW/'direct_target_continuation_v1/fit/normalization.npz'")
replace('def paths():','def paths(physics=None):')
replace("physics=NEW/'direct_target_independent_physics_v1/report.json'","physics=physics if physics is not None else NEW/'direct_target_fp64_independent_physics_v1/report.json'")
replace('def freeze():','def freeze(physics,physics_sha):')
replace("p=paths();assert sha(p['trace'])=='449a31d17aa7df6b23c20d6b0394237fc36b97d779f46544e6375975e0cdf427'","p=paths(physics);assert sha(p['trace'])=='c5829d6e54bc5791bf9e9409d055496f7e94402c9af8e39ec83cf3b654ac4cae'")
replace("assert sha(p['physics'])=='8538c0187f881a0662eaa2ee0f970bc69881921a24709575c571fec2b333db8c'","assert sha(p['physics'])==physics_sha\n    from release_checks import release_paths\n    p.update(release_paths(NEW))")
replace("req=read(BASE/'request.json');reqsha=sha(BASE/'request.json');assert sha(__file__)==req['source_sha256']","req=read(BASE/'request.json');reqsha=sha(BASE/'request.json');assert sha(__file__)==req['source_sha256']\n    from release_checks import check_release")
replace("p={k:local(v) for k,v in req['paths'].items()};a=load(p['trace']);r=read(p['report']);cap=load(p['failure'])","p={k:local(v) for k,v in req['paths'].items()};release=check_release(p)\n    a=load(p['trace']);r=read(p['report']);cap=load(p['failure'])")
for number,value in [('3158','2916'),('327','303'),('317','293'),('316','292'),('315','291'),('66','42')]:
 before=text;text,n=re.subn(r'\b'+number+r'\b',value,text);assert n>0;changes.append({'numeric_token':number,'replacement':value,'count':n})
replace('np.array([8],np.int64)','np.array([6],np.int64)')
replace("report=dict(passed=True,moving_controls=42,actual_controls=292,exact_checks=len(checks),checks=checks,","report=dict(passed=True,moving_controls=42,actual_controls=292,original_requested_controls=1569,original_lifecycle_completed=False,release_subjects=release,exact_checks=len(checks),checks=checks,")
replace("if sys.argv[1:] == ['--freeze']:freeze()\n    elif sys.argv[1:] == ['--run']:audit()\n    else:raise SystemExit('Use --freeze or --run')", "if sys.argv[1:] == ['--run']:audit()\n    else:\n        import argparse\n        parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true',required=True);parser.add_argument('--physics',type=Path,required=True);parser.add_argument('--physics-sha',required=True)\n        args=parser.parse_args();freeze(args.physics,args.physics_sha)")
ast.parse(text)
for name,content in [('audit_original.py',old),('audit_saved.py',text)]:
 with (B/name).open('x',encoding='utf-8') as f:f.write(content)
(B/'source_derivation.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='preserved_original',tofile='FP64_actual42_saved_audit')))
(B/'source_derivation.json').write_text(json.dumps({'original':str(oldpath),'original_sha256':digest,'actual_sha256':hashlib.sha256((B/'audit_saved.py').read_bytes()).hexdigest(),'changes':changes,'pure_observation_and_output_history_arithmetic_unchanged':True,'model_calls':0,'native_steps':0},indent=2)+'\n')
print('source preparation only')
