"""Source-only review receipt and generated PowerShell parser fixtures."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=NEW/'direct_target_student_evaluation_v1';OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={}
def bind(p):p=Path(p);pins[p.as_posix()]=sha(p);return p
for name in ('freeze_final_package.py','prepare_bound_launcher.py','diagnostic_verdict.py','test_launch_helpers.py'):
    p=bind(BASE/name);ast.parse(p.read_text())
test=bind(BASE/'launch_helper_tests_final_v2.xml');suite=ET.parse(test).getroot().find('testsuite')
assert suite.attrib['tests']=='21' and all(suite.attrib[k]=='0' for k in ('errors','failures','skipped'))
prior=bind(NEW/'direct_target_evaluation_source_review_v1/review.json')
assert sha(prior)=='01c11ab4d4cd06a914ccc29a4b9bced56f625dcaedfdc5605c9d81c2543e0529'
prep=bind(BASE/'source_preparation.json');p=json.loads(prep.read_text())
assert sha(prep)=='7391598abf062789b8daae00d85076470b947c0b3036242eccd8b2f5b36bab5f'
for name,digest in p['source_sha256'].items():assert sha(bind(BASE/'source_draft_v1'/name))==digest,name
spec=importlib.util.spec_from_file_location('reviewed_pure_launcher',BASE/'prepare_bound_launcher.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for mode in ('witness','evaluation'):
    args=module.wsl_arguments(BASE,mode)
    assert args[2:4]==['--cd','/']
    assert args[-1].endswith('/source_draft_v1/'+('head_activation_witness.py' if mode=='witness' else 'evaluate_direct_target_student.py'))
    folder=BASE/(mode+'_process')
    for suffix,text in [('run',module.run_text(BASE,folder,mode,'a'*64)),('durable',module.durable_text(folder))]:
        target=OUT/(mode+'_'+suffix+'_parser_fixture.ps1')
        with target.open('x',encoding='utf-8') as f:f.write(text)
        bind(target)
bind(__file__)
with (OUT/'source_check.json').open('x',encoding='utf-8') as f:
    json.dump(dict(source_checks_passed=True,source_sha256=pins,existing_tests=21,task_model_calls=0,native_steps=0),f,indent=2)
print(json.dumps({'source_checks_passed':True,'pins':len(pins),'generated_templates':4,'models':0,'native':0}))
