"""Preserve established saved-audit supervisor; edit metadata literals only."""
from pathlib import Path
import ast,difflib,json,hashlib
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_width512_fit_independent_v1'
changes={}
for name in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py'):
    before=(OLD/name).read_bytes();text=before.decode('utf-8')
    if name=='prepare_audit_request.py':
        assert text.count('direct_target_causal_width512_student_v1')==1
        text=text.replace('direct_target_causal_width512_student_v1','direct_target_width251_student_v1')
        text=text.replace("report['ordinary_final_step']==81000 and report['optimizer_step']==16000", "report['ordinary_final_step']==91000 and report['optimizer_step']==26000")
    text=text.replace('saved_width512_warm_only','saved_width251_recovery_warm_only')
    after=text.encode();(BASE/name).write_bytes(after)
    if name.endswith('.py'):ast.parse(text)
    changes[name]=dict(old_sha256=hashlib.sha256(before).hexdigest(),new_sha256=hashlib.sha256(after).hexdigest(),byte_exact=before==after)
    if before!=after:(BASE/(name+'.diff')).write_text(''.join(difflib.unified_diff(before.decode().splitlines(True),text.splitlines(True),fromfile='old/'+name,tofile='new/'+name)))
(BASE/'helper_derivation.json').write_text(json.dumps(changes,indent=2)+'\n')
print(json.dumps(changes))
