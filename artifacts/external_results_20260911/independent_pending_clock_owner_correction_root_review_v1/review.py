"""Review dispatch-only completion correction; no actual audit or task work."""
import ast,copy,hashlib,importlib.util,json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'independent_plant_pending_publication_saved_actual_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
old=BASE/'verify_completion.py';new=BASE/'verify_completion_dispatch_v2.py'
assert sha(old)=='2ad352000d8435b068ec2009b7bd5204a161feaa13fc96c4c0f8eac4d07a94e7'
assert sha(new)=='84aae1167a4e438a51a17428e4e81283e67319d917af3cffba06d1e05e48ed0a'
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
before,after=functions(old),functions(new)
assert set(after)==set(before)|{'validate_dispatch_correction'}
for name in before:
    if name!='main':assert before[name]==after[name],name
derive=ast.parse((BASE/'derive_owner_dispatch_v2.py').read_text())
check=next(ast.literal_eval(n.value) for n in derive.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECK' for t in n.targets))
text=old.read_text()
assert text.count("BASE/'dispatch.json'")==2
text=text.replace("BASE/'dispatch.json'","BASE/'dispatch_v2.json'")
text=text.replace('def main():',check+'def main():')
needle="    assert start['wrapper_pid']==child['wrapper_pid']==end['wrapper_pid']==dispatch['wrapper_pid']"
assert text.count(needle)==1;text=text.replace(needle,"    correction=validate_dispatch_correction(dispatch)\n"+needle)
needle="    output_paths+=list((BASE/'process_v1').glob('*.json'))"
assert text.count(needle)==1
text=text.replace(needle,"    output_paths += [BASE/'dispatch.json',BASE/'verify_completion.py',BASE/'derive_owner_dispatch_v2.py',BASE/'owner_dispatch_v2.diff'] + list((BASE/'preflight_dispatch_failure_v1').glob('*'))\n"+needle)
text=text.replace('    result=dict(completion_accounting_passed=True','    result=dict(dispatch_correction=correction,completion_accounting_passed=True')
text=text.replace("BASE/'owner_completion.json'","BASE/'owner_completion_dispatch_v2.json'")
assert ast.dump(ast.parse(text),include_attributes=False)==ast.dump(ast.parse(new.read_text()),include_attributes=False)
assert all(p.is_file() for p in (BASE/'preflight_dispatch_failure_v1').glob('*'))
spec=importlib.util.spec_from_file_location('owner_dispatch_review',new)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
actual=read(BASE/'dispatch_v2.json');correct=module.validate_dispatch_correction(actual)
assert correct['prior_actual_audit_calls']==0 and actual['wrapper_pid']==21588
mutations=[('prior_actual_audit_calls',1),('wrapper_pid',27700),('automatic_retry',True),('handle_acquired',False),
           ('original_dispatch_sha256','0'*64),('utc','2020-01-01T00:00:00Z')]
for key,value in mutations:
    bad=copy.deepcopy(actual);bad[key]=value
    try:module.validate_dispatch_correction(bad)
    except AssertionError:pass
    else:raise AssertionError('Accepted corrupt '+key)
bad=copy.deepcopy(actual);bad['exact_arguments'][-1]='0'*64
try:module.validate_dispatch_correction(bad)
except AssertionError:pass
else:raise AssertionError('Accepted wrong receipt argument')
assert not (BASE/'owner_completion_dispatch_v2.json').exists()
report=dict(passed=True,source_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 original_helper_sha256=sha(old),corrected_helper_sha256=sha(new),actual_dispatch_sha256=sha(BASE/'dispatch_v2.json'),
 correction=correct,unchanged_original_functions=True,main_changes_limited_to_dispatch_linkage_and_additive_output_provenance=True,
 metadata_checks=8,actual_audit_executed=False,actual_owner_verifier_executed=False,
 model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,writer_sha256=sha(__file__),
 reviewed_semantics=['Original27700 launch stopped during mandatory argument binding; saved absence/no script or result evidence bound.',
 'Corrected21588 launch exact noninteractive command/receipt argument and same reviewed request/launcher/clearance.',
 'All original raw exit, process chain/absence, source/pre/post pins, comparisons and truthful evidence outcome checks retained.',
 'Original dispatch/helper retained; new owner file includes both attempts and marks actual audit not repeated.'])
path=OUT/'review.json'
with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'path':path.as_posix(),'sha256':sha(path),'passed':True,'metadata_checks':8}))
