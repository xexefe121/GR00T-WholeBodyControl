"""Preserve prepared v1; make only fault-publication immutability corrections."""
from pathlib import Path
import hashlib,json,shutil
BASE=Path(__file__).resolve().parent
old=json.loads((BASE/'preparation_report.json').read_text())
for name,digest in old['source_sha256'].items():
    assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest
source=BASE/'source_draft_v2';assert not source.exists()
shutil.copytree(BASE/'source_draft',source,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
path=source/'clock_core.py';text=path.read_text()
edits=[
    ('import json\n','import json\nfrom types import MappingProxyType\n'),
    ('self.failure = dict(reason=reason, physics=self.returned, **data)',
     'self.failure = MappingProxyType(dict(reason=reason, physics=self.returned, **data))'),
    ("self.input_fault = dict(reason='COMMAND_DEADLINE_MISSED', control=control,\n                                            nominal_deadline=self.deadline(self.returned))",
     "self.input_fault = MappingProxyType(dict(reason='COMMAND_DEADLINE_MISSED', control=control,\n                                            nominal_deadline=self.deadline(self.returned)))"),
    ("self.first_deadline_failure = dict(index=index, nominal_end=self.deadline(index + 1),\n                    observed_ns=start, phase='WAKE_ALREADY_OVERDUE', lateness_ns=start-self.deadline(index+1))",
     "self.first_deadline_failure = MappingProxyType(dict(index=index, nominal_end=self.deadline(index + 1),\n                    observed_ns=start, phase='WAKE_ALREADY_OVERDUE', lateness_ns=start-self.deadline(index+1)))"),
    ("self.first_deadline_failure = dict(index=index, nominal_end=self.deadline(index + 1),\n                                                       actual_end=finish, lateness_ns=lateness)",
     "self.first_deadline_failure = MappingProxyType(dict(index=index, nominal_end=self.deadline(index + 1),\n                                                       actual_end=finish, lateness_ns=lateness))"),
    ('first_deadline_failure=self.first_deadline_failure,','first_deadline_failure=dict(self.first_deadline_failure) if self.first_deadline_failure is not None else None,'),
    ('epoch_ns=self.epoch_ns, failure=self.failure, input_fault=self.input_fault,',
     'epoch_ns=self.epoch_ns, failure=dict(self.failure) if self.failure is not None else None,\n            input_fault=dict(self.input_fault) if self.input_fault is not None else None,')]
for before,after in edits:
    assert text.count(before)==1,before
    text=text.replace(before,after)
path.write_text(text,encoding='utf-8')
test=source/'test_foundation.py'
with test.open('a',encoding='utf-8') as f:
    f.write('''

def test_logger_cannot_modify_latched_fault_or_timing_records():
    p=plant(50);p.tick();p.clock.advance(23_000_000);run(p)
    original=p.summary()
    with pytest.raises(TypeError):p.first_deadline_failure['index']=999
    with pytest.raises(TypeError):p.input_fault['control']=999
    copy=p.summary();copy['first_deadline_failure']['index']=999;copy['input_fault']['control']=999
    assert p.summary()==original
    failed=plant(50,max_debt_steps=1);failed.tick();failed.clock.advance(23_000_000);failed.tick()
    original_failure=dict(failed.failure)
    with pytest.raises(TypeError):failed.failure['reason']='erased'
    failed.summary()['failure']['reason']='erased'
    assert dict(failed.failure)==original_failure
''')
readme=(BASE/'README.md').read_text().replace('foundation — source preparation only','foundation v2 — source preparation only')
readme=readme.replace('from `source_draft`','from `source_draft_v2`').replace('40 passing tests','41 passing tests')
readme+='\nV2 preserves the complete v1 source/report/test evidence. It makes latched input, timing and failure records immutable, and returns copies in summaries so a logger cannot alter plant fault state. Scheduling, admission, history and step arithmetic are unchanged.\n'
(BASE/'README_V2.md').write_text(readme,encoding='utf-8')
recorder=(BASE/'record_preparation.py').read_text().replace("SOURCE=BASE/'source_draft'","SOURCE=BASE/'source_draft_v2'")
recorder=recorder.replace("BASE/'tests_final.xml'","BASE/'tests_v2_final.xml'").replace("==40","==41").replace('tests=40','tests=41').replace("tests=40","tests=41")
recorder=recorder.replace("label+'_fake_example.json'","label+'_fake_example_v2.json'")
recorder=recorder.replace("BASE/'README.md'","BASE/'README_V2.md'")
recorder=recorder.replace("BASE/'preparation_report.json'","BASE/'preparation_report_v2.json'")
recorder=recorder.replace("Path(__file__),design/'NOTE.md'", "Path(__file__),BASE/'preparation_report.json',BASE/'prepare_immutable_faults_v2.py',design/'NOTE.md'")
(BASE/'record_preparation_v2.py').write_text(recorder,encoding='utf-8')
print('Prepared v2 without changing v1 source/evidence.')
