"""Record source preparation and deterministic fake-only example ledgers."""
import ast
import base64
from dataclasses import asdict,is_dataclass
import hashlib,json,sys
from pathlib import Path
import xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SOURCE=BASE/'source_draft_v2'
sys.path.insert(0,str(SOURCE))
from test_foundation import plant,run

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def serial(value):
    if is_dataclass(value):return serial(asdict(value))
    if isinstance(value,bytes):return dict(base64=base64.b64encode(value).decode())
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value
def write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(serial(data),f,indent=2,allow_nan=False);f.write('\n')

test=ET.parse(BASE/'tests_v2_final.xml').getroot().find('testsuite')
assert int(test.attrib['tests'])==41 and all(int(test.attrib[k])==0 for k in ('errors','failures','skipped'))
source_hashes={p.relative_to(BASE).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
for path in SOURCE.glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
design=NEW/'independent_plant_clock_design_v1'
design_pins=read(design/'source_pins.json')
assert sha(design/'NOTE.md')==design_pins['note_sha256']=='37845f829d98555e2ace39fbec43e0375d0c2915d076e602670dfff72b88d401'
runtime_checks={}
for name,expected in design_pins['source_sha256'].items():
    if name.endswith('.py'):
        actual=sha(Path(name));assert actual==expected,name
        runtime_checks[name]=dict(expected=expected,actual=actual,unchanged=True)
examples={}
for label in ('hung_worker','plant_stall23ms'):
    p=plant(50)
    if label=='hung_worker':
        p.jobs.slots[1].lock.held=True;p.results.slots[2].lock.held=True
    else:
        p.tick();p.clock.advance(23_000_000)
    result=run(p)
    assert result['returned']==result['captured']==result['verified']==50
    assert result['timing_passed']==(label=='hung_worker')
    path=BASE/(label+'_fake_example_v2.json')
    write(path,dict(fake_only=True,summary=result,steps=p.step_records.records(),controls=p.control_records.records(),
        events=[json.loads(v) for v in p.events.records()],actual_native_steps=0,controller_calls=0))
    examples[path.name]=sha(path)
evidence={str(p).replace('\\','/'):sha(p) for p in (
    BASE/'README_V2.md',BASE/'tests_v1.xml',BASE/'tests_v2.xml',BASE/'tests_v3.xml',BASE/'tests_v2_final.xml',
    Path(__file__),BASE/'preparation_report.json',BASE/'prepare_immutable_faults_v2.py',design/'NOTE.md',design/'source_pins.json')}
report=dict(passed=True,source_preparation_only=True,tests=41,failures=0,errors=0,skips=0,
    source_sha256=source_hashes,evidence_sha256=evidence,example_sha256=examples,
    original_runtime_source_checks=runtime_checks,all_original_runtime_sources_unchanged=True,
    step_ns=2000000,control_steps=10,history_four_lag_source_byte_parity=True,
    source_identity_gaps_retained=True,source_window_completeness_implemented=False,
    actual_native_steps=0,ORT_calls=0,controller_calls=0,optimizer_updates=0,
    actual_spawned_process_tests=0,actual_process_shared_storage=False,new_runtime_installed=False,
    fake_example_steps=100,real_process_timing_qualified=False,balance_qualified=False,
    limitations_path=(BASE/'README_V2.md').as_posix())
write(BASE/'preparation_report_v2.json',report)
print(json.dumps(dict(report_sha256=sha(BASE/'preparation_report_v2.json'),source_sha256=source_hashes,tests=41)))
