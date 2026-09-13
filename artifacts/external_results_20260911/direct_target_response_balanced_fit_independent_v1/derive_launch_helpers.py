"""Copy the qualified saved-audit supervisor with only subject/path adaptations."""
from pathlib import Path
import ast,difflib,hashlib,json
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_context_pair_fit_independent_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
mapping={'run_audit_durable_v3.ps1':'run_audit_durable.ps1','freeze_launch_v3.py':'freeze_launch.py','verify_completion_v3.py':'verify_completion.py'}
patch=[];lineage={}
for oldname,newname in mapping.items():
    old=(OLD/oldname).read_text();new=old
    for a,b in [('saved_matched_context_pair_only','saved_response_balanced_warm_only'),('source_draft_v3','source_prepared_v1'),
        ('audit_saved_pair.py','audit_saved_warm.py'),('run_audit_durable_v3.ps1','run_audit_durable.ps1'),
        ('prepare_audit_request_v3.py','prepare_audit_request.py'),('source_preparation_v3.json','source_preparation.json'),
        ('verify_completion_v3.py','verify_completion.py')]:new=new.replace(a,b)
    if newname=='freeze_launch.py':
        anchor="    owner=request['subjects']['owner_completion'];assert sha(owner['path'])==owner['sha256']\n"
        assert new.count(anchor)==1
        new=new.replace(anchor,anchor+"    review=read(request['source_review']['path']);assert sha(request['source_review']['path'])==request['source_review']['sha256'] and review[request['source_review']['pass_field']] is True\n    for name in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py'):assert review['helper_sha256'][name]==sha(BASE/name)\n")
    if newname.endswith('.py'):ast.parse(new)
    with (BASE/newname).open('x',encoding='utf-8') as f:f.write(new)
    patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='qualified/'+oldname,tofile=newname))
    lineage[newname]=dict(source_path=(OLD/oldname).as_posix(),source_sha256=sha(OLD/oldname),sha256=sha(BASE/newname))
with (BASE/'launch_helper_delta.patch').open('x') as f:f.write(''.join(patch))
with (BASE/'launch_helper_derivation.json').open('x') as f:json.dump(lineage,f,indent=2)
print('Prepared three launch helpers; no request or execution.')
