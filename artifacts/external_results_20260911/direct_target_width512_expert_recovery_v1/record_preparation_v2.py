"""Preserve v1; freeze metadata-only phase correction without task execution."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def subject(p):return dict(path=str(p).replace('\\','/'),sha256=sha(p))

def main():
    old=BASE/'source_preparation.json';receipt=json.loads(old.read_text())
    draft=BASE/'source_draft_v1';previous=BASE/'source_prepared_v1';target=BASE/'source_prepared_v2'
    filename='run_width251_actual_oracle.py'
    expected=(previous/filename).read_text().replace("save_work('first_query_complete')","save_work('initial_seed_certification_complete')")
    line="                        incoming_history=s['snapshot']['history_flat'].copy(),proposal_before_preview=np.asarray(True))"
    expected=expected.replace(line,line+"\n                    save_work('first_optimized_target_ready')")
    assert expected==(draft/filename).read_text()
    tests=ET.parse(BASE/'phase_tests_v2.xml').getroot().findall('testsuite')
    assert sum(int(v.attrib['tests']) for v in tests)==1 and all(int(v.attrib['failures'])==int(v.attrib['errors'])==0 for v in tests)
    target.mkdir(exist_ok=False);sources={}
    for name,oldsha in receipt['source_sha256'].items():
        p=draft/name
        if name!=filename:assert sha(p)==oldsha
        out=target/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes());sources[name]=sha(out)
    receipt.update(source_directory=str(target).replace('\\','/'),source_sha256=sources,
        previous_preparation=subject(old),metadata_only_correction=dict(
            changed_source=filename,seed_phase='initial_seed_certification_complete',optimized_phase='first_optimized_target_ready',
            mathematical_source_changed=False,task_calls=0,additional_saved_snapshot_per_first_optimized_target=1),
        additional_tests=dict(passed=True,tests=1,receipt=subject(BASE/'phase_tests_v2.xml'),
                              source=subject(BASE/'test_phase_labels_v2.py')),
        revision_source=subject(Path(__file__)))
    with (BASE/'source_preparation_v2.json').open('x',newline='\n') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(subject(BASE/'source_preparation_v2.json')))

if __name__=='__main__':main()
