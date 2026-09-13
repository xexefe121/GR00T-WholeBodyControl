from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
tree=ET.parse(BASE/'launch_helper_tests_v3.xml').getroot()
suites=[tree] if tree.tag=='testsuite' else list(tree.iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==41 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
derivation=read(BASE/'launch_helper_derivation_v2.json')
for name,digest in derivation['helper_sha256'].items():assert sha(BASE/name)==digest
for name,digest in derivation['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest
assert read(BASE/'source_preparation.json')['source_sha256']==derivation['source_sha256']
record=dict(derivation,source_preparation_passed=True,synthetic_tests=41,
    source_preparation_subject=dict(path=(BASE/'source_preparation.json').as_posix(),sha256=sha(BASE/'source_preparation.json')),
    subjects={name:dict(path=(BASE/name).as_posix(),sha256=sha(BASE/name)) for name in
        ('launch_helper_derivation_v2.json','launch_helper_tests_v3.xml','powershell_path_normalization_v1.json',
         'amend_fit_namespace_v2.py','record_helpers_v2.py')},actual_launches=0)
with (BASE/'launch_helper_preparation_v2.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(dict(helper_preparation_sha256=sha(BASE/'launch_helper_preparation_v2.json'),helpers=8,tests=41)))
