import hashlib,json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'independent_plant_timing_instrumentation_v1'
SOURCE=BASE/'source_draft_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
prep=BASE/'source_preparation_v2.json'
assert sha(prep)=='4c0b40ae3decb12af8165b52fdbfb578d3edc946b2d286cc3112fc7fb2d6af19'
r=read(prep)
assert r['passed'] and r['tests']==31 and r['failures']==r['errors']==r['skips']==0
for name,digest in r['source_sha256'].items():assert sha(SOURCE/name)==digest
for name,digest in r['evidence_sha256'].items():assert sha(BASE/name)==digest
assert sha(SOURCE/'timing_probe.py')=='a41ef15e1e47afc5d6fb3c0ab27a203208b1c66c10559a1ae8792e6e0dbe57f4'
with (OUT/'tests.log').open('x',encoding='utf-8') as f:
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(SOURCE),'-p','test*.py','-v'],cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT)
assert result.returncode==0
log=(OUT/'tests.log').read_text()
assert 'Ran 31 tests' in log and log.rstrip().endswith('OK')
for name,digest in r['source_sha256'].items():assert sha(SOURCE/name)==digest
review=dict(passed=True,source_only=True,preparation_sha256=sha(prep),source_sha256=r['source_sha256'],
 independent_synthetic_tests=31,tests_sha256=sha(OUT/'tests.log'),writer_sha256=sha(__file__),
 reviewed=['Preallocated bounded span/GC tables and no overwrite after capacity failure',
 'Known return flag precedes end clocks; original exception preservation remains integration responsibility',
 'Local wall brackets and paired same-thread spans permit valid nested GC and foreign-thread overlap',
 'Prior global per-read monotonic thresholds removed; backward local/paired/root clocks still rejected',
 'Foreign-thread inside_probe must be interpreted as overlap, not proof of same-thread interruption'],
 actual_probe_clock_calls=0,actual_gc_callbacks_installed=0,native_steps=0,model_calls=0,
 runtime_integration_reviewed=False,actual_clock_selected=False,overhead_measured=False,component_qualified=False)
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'tests':31,'sha256':sha(OUT/'review.json')}))
