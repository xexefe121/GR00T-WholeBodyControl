"""Freeze the reviewed existing fit validator; no task arrays or models loaded."""
import ast,hashlib,json,shutil
import xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_width512_fit_independent_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
for file,count in [('root_source_tests_v2.xml',74),('root_process_tests_v2.xml',12)]:
    root=ET.parse(BASE/file).getroot();suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)==count
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
source=BASE/'source_draft_v1';dest=BASE/'source_prepared_v1';dest.mkdir(exist_ok=False)
files={p.name:sha(p) for p in source.glob('*.py')}
old=read(OLD/'source_preparation.json')
for name in ('audit_balanced_math.py','audit_context_math.py','audit_full_state_math.py','audit_graph.py','audit_math.py','audit_restoration.py','audit_width_math.py'):
    assert files[name]==old['source_sha256'][name]
for name,digest in files.items():
    ast.parse((source/name).read_text(encoding='utf-8-sig'));shutil.copyfile(source/name,dest/name);assert sha(dest/name)==digest
helpers=read(BASE/'helper_derivation.json')
for name,value in helpers.items():assert sha(BASE/name)==value['new_sha256']
result=dict(source_preparation_passed=True,source_directory=dest.as_posix(),experiment=(BASE.parent/'direct_target_width251_student_v1').as_posix(),
    source_sha256=files,helper_sha256={n:sha(BASE/n) for n in helpers},tests=86,
    evidence_sha256={n:sha(BASE/n) for n in ('root_source_tests_v2.xml','root_process_tests_v2.xml','main_derivation.patch','helper_derivation.json')},
    model_calls=0,native_steps=0,task_array_reads=0,actual_audit_executed=False,
    corrections=['Inherited test fixtures updated for recovery roles.',
      'Process checker matches actual owner dispatch-attempt and four log subjects; omitted or altered evidence rejected.'],
    independent_unchanged_math_modules=7)
write(BASE/'source_preparation.json',result)
write(BASE/'source_root_review.json',dict(source_review_pass=True,passed=True,**{k:result[k] for k in ('source_sha256','helper_sha256','tests','corrections')},
    source_preparation_sha256=sha(BASE/'source_preparation.json'),model_calls=0,native_steps=0,physical_qualification=False))
print(json.dumps({'source_count':len(files),'review_sha256':sha(BASE/'source_root_review.json'),'source_preparation_sha256':sha(BASE/'source_preparation.json')}))
