import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
xml=ET.parse(BASE/'launch_helper_tests_v3.xml').getroot()
suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites)==41
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
case=[c for c in xml.iter('testcase') if c.attrib['name']=='test_actual_powershell_generated_path_normalization']
assert len(case)==1 and not list(case[0])
path_record=dict(passed=True,kind='completed_actual_PowerShell_assertion_in_pinned_pytest',
 test_file_sha256=sha(BASE/'test_release_helpers.py'),test_xml_sha256=sha(BASE/'launch_helper_tests_v3.xml'),
 test_name=case[0].attrib['name'],actual_task_release_reads=0,actual_launches=0)
with (BASE/'powershell_path_normalization_v1.json').open('x',encoding='utf-8') as f:json.dump(path_record,f,indent=2);f.write('\n')
derivation=read(BASE/'launch_helper_derivation.json');prep=read(BASE/'source_preparation.json')
for n,h in derivation['helper_sha256'].items():assert sha(BASE/n)==h
for n,h in prep['source_sha256'].items():assert sha(BASE/'source_draft_v1'/n)==h
record=dict(derivation,source_preparation_passed=True,synthetic_tests=41,source_sha256=prep['source_sha256'],
 source_preparation_subject=dict(path=(BASE/'source_preparation.json').as_posix(),sha256=sha(BASE/'source_preparation.json')),
 subjects={n:dict(path=(BASE/n).as_posix(),sha256=sha(BASE/n)) for n in
 ('launch_helper_derivation.json','launch_helper_tests_v3.xml','powershell_path_normalization_v1.json','prepare_launch_helpers.py','record_launch_helpers.py')},
 actual_launches=0)
path=BASE/'launch_helper_preparation_v2.json'
with path.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'helper_preparation_sha256':sha(path),'helpers':8,'tests':41}))
