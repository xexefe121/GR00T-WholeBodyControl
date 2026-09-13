"""Bind helper derivation and synthetic evidence; no release or model calls."""
from pathlib import Path
import hashlib,json,subprocess,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
derivation=read(BASE/'launch_helper_derivation.json')
for name,digest in derivation['helper_sha256'].items():assert sha(BASE/name)==digest
for name,digest in derivation['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest
assert read(BASE/'source_preparation.json')['source_sha256']==derivation['source_sha256']
tree=ET.parse(BASE/'launch_helper_tests_v2.xml').getroot()
suites=[tree] if tree.tag=='testsuite' else list(tree.iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==41
assert not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
from prepare_bound_launcher import durable_text
text=durable_text(BASE/'synthetic_process')
marker='$receiptPath.Replace('
expression=marker+text.split(marker,1)[1].split(')',1)[0]+')'
command="$receiptPath='E:\\synthetic\\evaluation_process\\launch_receipt.json'; $actual="+expression+"; if ($actual -cne 'E:/synthetic/evaluation_process/launch_receipt.json') { throw 'normalization failed' }; Write-Output $actual"
result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
assert result.stdout.strip()=='E:/synthetic/evaluation_process/launch_receipt.json'
normalization=dict(passed=True,generated_expression=expression,actual_output=result.stdout.strip(),
    prepare_bound_launcher_sha256=sha(BASE/'prepare_bound_launcher.py'),task_model_calls=0,native_steps=0)
with (BASE/'powershell_path_normalization_v1.json').open('x',encoding='utf-8') as stream:json.dump(normalization,stream,indent=2);stream.write('\n')
receipt=dict(derivation,source_preparation_passed=True,synthetic_tests=41,
    source_preparation_subject=dict(path=(BASE/'source_preparation.json').as_posix(),sha256=sha(BASE/'source_preparation.json')),
    subjects={name:dict(path=(BASE/name).as_posix(),sha256=sha(BASE/name)) for name in
        ('launch_helper_derivation.json','launch_helper_tests_v2.xml','powershell_path_normalization_v1.json',
         'prepare_launch_helpers.py','record_launch_helper_preparation.py')},
    actual_launches=0,root_fit_runtime_untouched=True)
with (BASE/'launch_helper_preparation_v1.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(dict(helper_preparation_sha256=sha(BASE/'launch_helper_preparation_v1.json'),helpers=8,tests=41,model_calls=0,native_steps=0)))
