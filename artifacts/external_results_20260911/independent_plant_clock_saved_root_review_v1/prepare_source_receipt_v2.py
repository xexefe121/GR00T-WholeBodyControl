"""Narrow v2 source/test hashing only; no actual task arrays or audit execution."""
import hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    prior=json.loads((BASE/'source_preparation_audit_v1.json').read_text());source=BASE/'source_audit_v2'
    for name,digest in prior['source_sha256'].items():assert sha(BASE/'source_audit_v1'/name)==digest
    for name,digest in prior['original_root_files_sha256'].items():assert sha(BASE/name)==digest
    current={p.name:sha(p) for p in sorted(source.glob('*.py'))};assert set(current)==set(prior['source_sha256'])
    changed=[name for name in current if current[name]!=prior['source_sha256'][name]]
    assert changed==['audit_clock.py','clock_saved_math.py','test_clock_saved_math.py']
    xml=BASE/'audit_synthetic_tests_final_v2.xml';tree=ET.parse(xml);assert len(tree.findall('.//testcase'))==43
    assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
    result=dict(kind='independent_clock_saved_auditor_source_preparation',version=2,preparation_only=True,actual_saved_audit_selected=False,
        source_directory=source.as_posix(),source_sha256=current,original_root_files_sha256=prior['original_root_files_sha256'],
        preserved_v1_receipt_sha256=sha(BASE/'source_preparation_audit_v1.json'),changed_from_v1=changed,unchanged_v1_files=9,
        tests=dict(passed=43,failed=0,skipped=0,path=xml.as_posix(),sha256=sha(xml),python=sys.version,numpy=np.__version__),
        documentation_sha256=sha(BASE/'AUDIT_PREPARATION_V2.md'),preparation_source_sha256=sha(Path(__file__)),
        actual_task_array_evaluations=0,native_steps=0,model_calls=0,optimizer_updates=0,worker_processes=0,actual_request_created=False,
        correction='Nonfinite actual captured state/force remains failure evidence; original initial step-zero force exception is unchanged.')
    path=BASE/'source_preparation_audit_v2.json'
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),source_files=12,tests=43)))
if __name__=='__main__':main()
