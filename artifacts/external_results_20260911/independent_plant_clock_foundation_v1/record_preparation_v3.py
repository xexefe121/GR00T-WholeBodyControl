"""Record the reviewed-probe corrections; all stepper activity remains fake."""
from pathlib import Path
from dataclasses import asdict, is_dataclass, replace
import ast
import base64
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SOURCE=BASE/'source_draft_v3'
sys.path.insert(0,str(SOURCE))
from test_foundation import plant,run


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def serial(value):
    if is_dataclass(value):return serial(asdict(value))
    if isinstance(value,bytes):return dict(base64=base64.b64encode(value).decode())
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value
def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(serial(value),stream,indent=2,allow_nan=False);stream.write('\n')

old=read(BASE/'preparation_report_v2.json')
preservation={}
for field in ('source_sha256','evidence_sha256','example_sha256'):
    for name,expected in old[field].items():
        path=Path(name)
        if not path.is_absolute():path=BASE/path
        actual=sha(path);assert actual==expected,str(path)
        preservation[path.as_posix()]=dict(expected=expected,actual=actual,unchanged=True)
test=ET.parse(BASE/'tests_v3_capture_clock_final.xml').getroot().find('testsuite')
assert int(test.attrib['tests'])==70 and all(int(test.attrib[k])==0 for k in ('errors','failures','skipped'))
sources={p.relative_to(BASE).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
for path in SOURCE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
assert sha(SOURCE/'history.py')==sha(BASE/'source_draft_v2/history.py')
assert sha(SOURCE/'mailbox.py')==sha(BASE/'source_draft_v2/mailbox.py')
design=NEW/'independent_plant_clock_design_v1'
runtime_checks={}
for name,expected in read(design/'source_pins.json')['source_sha256'].items():
    if name.endswith('.py'):
        actual=sha(name);assert actual==expected,name
        runtime_checks[name]=dict(expected=expected,actual=actual,unchanged=True)
examples={}
for label in ('hung_worker','plant_stall23ms'):
    p=plant(50)
    if label=='hung_worker':
        p.jobs.slots[1].lock.held=True;p.results.slots[2].lock.held=True
    else:p.tick();p.clock.advance(23_000_000)
    summary=run(p)
    assert summary['returned']==summary['captured']==summary['verified']==50
    assert summary['timing_passed']==(label=='hung_worker')
    path=BASE/(label+'_fake_example_v3.json')
    write(path,dict(fake_only=True,summary=summary,steps=p.step_records.records(),
                   controls=p.control_records.records(),events=[json.loads(v) for v in p.events.records()],
                   actual_native_steps=0,model_calls=0))
    examples[path.name]=sha(path)
probes={}
p=plant(1);issue={'message':['original']};p.stepper.verify_step=lambda capture:issue
p.tick();evidence=p.last_verifier_return;issue['message'][0]='mutated original'
assert p.last_verifier_return==evidence and p.captured==1 and p.verified==0
probes['mutable_verifier_issue']=dict(summary=p.summary(),return_evidence=evidence,
                                      last_valid_capture=p.last_capture)
p=plant(1);capture=p.stepper.capture_step;warning={'count':0}
p.stepper.capture_step=lambda:replace(capture(),warnings=(warning,))
p.tick();evidence=p.last_capture_return;warning['count']=99
assert p.last_capture_return==evidence and p.captured==0 and p.last_capture is None
probes['mutable_warning_capture']=dict(summary=p.summary(),return_evidence=evidence,
                                       last_valid_capture=p.last_capture)
p=plant(1,stepper={'duration':-1});p.tick()
assert not p.summary()['timing_passed'] and p.step_records.committed==0
probes['backward_finish_clock']=dict(summary=p.summary(),last_valid_capture=p.last_capture)
path=BASE/'root_probe_regressions_v3.json';write(path,dict(fake_steps=3,native_steps=0,model_calls=0,probes=probes))
examples[path.name]=sha(path)
evidence={p.as_posix():sha(p) for p in [Path(__file__),BASE/'README_V3.md',
    BASE/'tests_v3_capture_clock.xml',BASE/'tests_v3_capture_clock_final.xml',
    BASE/'preparation_report_v2.json',NEW/'independent_plant_clock_root_review_v1/findings.json',
    design/'NOTE.md',design/'source_pins.json']}
report=dict(passed=True,source_preparation_only=True,tests=70,failures=0,errors=0,skips=0,
    source_sha256=sources,evidence_sha256=evidence,example_sha256=examples,
    preserved_v2_checks=preservation,all_v2_evidence_preserved=True,
    original_runtime_source_checks=runtime_checks,all_original_runtime_sources_unchanged=True,
    verifier_contract='None or exact nonempty str, at most 1024 characters',
    capture_contract='exact CapturedStep; finite nonnegative float time; nonempty immutable state <=16384 bytes; finite float64[23] torque bytes; WarningLedger with exact eight-int32 count/lastinfo tuples',
    invalid_capture_return='stable owned typed diagnostic bytes, no captured/verified/committed credit or mutable last_capture alias',
    diagnostic_limits='depth8/items64/bytes-prefix4096; truncation/unsupported markers explicit, bytes size and full hash retained',
    clock_contract='all readings exact nonnegative monotonic integer ns; first clock fault latched; invalid current clock gives null current_debt',
    step_ns=2000000,control_steps=10,history_four_lag_source_byte_parity=True,
    source_identity_gaps_retained=True,source_window_completeness_implemented=False,
    actual_native_steps=0,ORT_calls=0,controller_calls=0,optimizer_updates=0,
    actual_spawned_process_tests=0,actual_process_shared_storage=False,new_runtime_installed=False,
    fake_example_steps=103,real_process_timing_qualified=False,balance_qualified=False,
    limitations_path=(BASE/'README_V3.md').as_posix())
for name,expected in sources.items():assert sha(BASE/name)==expected
write(BASE/'preparation_report_v3.json',report)
print(json.dumps(dict(report_sha256=sha(BASE/'preparation_report_v3.json'),source_sha256=sources,tests=70)))
