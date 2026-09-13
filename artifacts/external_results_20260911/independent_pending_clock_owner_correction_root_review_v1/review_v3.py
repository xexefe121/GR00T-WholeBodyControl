"""Lossless UTC ordering correction; saved metadata checks only."""
import ast,copy,hashlib,importlib.util,io,json,sys,unittest
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'independent_plant_pending_publication_saved_actual_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
v2=BASE/'verify_completion_dispatch_v2.py';v3=BASE/'verify_completion_dispatch_v3.py'
assert sha(v2)=='84aae1167a4e438a51a17428e4e81283e67319d917af3cffba06d1e05e48ed0a'
assert sha(v3)=='65d5847f7999bf63834ace87506bcb462ab728c2064ec0c0ac9d5ea513b7c0e4'
# Execute the original root source comparison up to, but not including, metadata validation.
prior=OUT/'review.py';prefix=prior.read_text().split("actual=read(BASE/'dispatch_v2.json')")[0]
exec(compile(prefix,str(prior),'exec'),{'__file__':str(prior)})
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
before,after=functions(v2),functions(v3)
assert set(after)==set(before)|{'utc_order_key'}
for name in before:
    if name not in ('validate_dispatch_correction','main'):assert before[name]==after[name],name
text=v2.read_text()
text=text.replace("assert datetime.fromisoformat(dispatch['utc'].replace('Z','+00:00'))>datetime.fromisoformat(stop['utc'].replace('Z','+00:00'))","assert utc_order_key(dispatch['utc'])>utc_order_key(stop['utc'])")
needle="    output_paths += [BASE/'dispatch.json'"
assert text.count(needle)==1
text=text.replace(needle,"    output_paths += [BASE/'verify_completion_dispatch_v2.py',BASE/'derive_owner_dispatch_v3.py',BASE/'owner_dispatch_v3.diff',BASE/'test_owner_timestamp.py',BASE/'owner_timestamp_tests.log']\n"+needle)
text=text.replace("BASE/'owner_completion_dispatch_v2.json'","BASE/'owner_completion_dispatch_v3.json'")
wanted={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
for name in ('validate_dispatch_correction','main'):assert wanted[name]==after[name]
spec=importlib.util.spec_from_file_location('owner_v3_root_review',v3)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
actual=read(BASE/'dispatch_v2.json');correction=mod.validate_dispatch_correction(actual)
assert actual['wrapper_pid']==21588 and correction['prior_actual_audit_calls']==0
for key,value in [('prior_actual_audit_calls',1),('wrapper_pid',27700),('automatic_retry',True),('handle_acquired',False),('original_dispatch_sha256','0'*64),('utc','2020-01-01T00:00:00Z')]:
    bad=copy.deepcopy(actual);bad[key]=value
    try:mod.validate_dispatch_correction(bad)
    except AssertionError:pass
    else:raise AssertionError('Accepted corrupt '+key)
bad=copy.deepcopy(actual);bad['exact_arguments'][-1]='0'*64
try:mod.validate_dispatch_correction(bad)
except AssertionError:pass
else:raise AssertionError('Accepted wrong receipt argument')
sys.path.insert(0,str(BASE))
suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_owner_timestamp.py')
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
with (OUT/'root_timestamp_tests.log').open('x',encoding='utf-8') as f:f.write(stream.getvalue())
assert result.wasSuccessful() and result.testsRun==5 and not result.skipped,stream.getvalue()
assert not (BASE/'owner_completion_dispatch_v3.json').exists()
report=dict(passed=True,source_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 original_helper_sha256=sha(BASE/'verify_completion.py'),preserved_v2_sha256=sha(v2),corrected_helper_sha256=sha(v3),
 original_root_checker_preserved=sha(prior),actual_dispatch_sha256=sha(BASE/'dispatch_v2.json'),correction=correction,
 unchanged_original_outcome_and_pin_functions=True,metadata_checks=8,root_timestamp_tests=5,
 root_timestamp_test_log_sha256=sha(OUT/'root_timestamp_tests.log'),writer_sha256=sha(__file__),
 actual_audit_executed=False,actual_owner_verifier_executed=False,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,
 reviewed_semantics=['V2 linkage checks preserved; only strict UTC whole-calendar-second plus integer-nanosecond ordering replaces incompatible Python3.10 fractional parser.',
 'Z/+00:00 only, optional1..9fraction digits, calendar validation, no floating point or fractional truncation.',
 'Original failed dispatch, corrected actual dispatch, all original raw exit/source/pre/post/output/PID checks retained; additive versioned owner.'])
path=OUT/'review_v3.json'
with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'path':path.as_posix(),'sha256':sha(path),'passed':True,'metadata_checks':8,'timestamp_tests':5}))
