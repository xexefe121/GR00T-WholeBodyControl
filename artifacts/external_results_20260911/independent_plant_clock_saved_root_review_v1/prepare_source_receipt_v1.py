"""Source/test hashing only. Does not inspect actual task arrays or run an audit."""
import hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    source=BASE/'source_audit_v1'
    original={name:sha(BASE/name) for name in ('clock_saved_math.py','clock_control_math.py','test_clock_saved_math.py','test_clock_control_math.py')}
    decoder=BASE.parent/'independent_native_stepper_saved_audit_v1/source_v2/saved_math.py'
    assert sha(decoder)==sha(source/'qualified_binary_math.py')
    xml=BASE/'audit_synthetic_tests_final_v1.xml';tree=ET.parse(xml)
    cases=tree.findall('.//testcase');assert len(cases)==40 and not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
    files={p.name:sha(p) for p in sorted(source.glob('*.py'))}
    receipt=dict(kind='independent_clock_saved_auditor_source_preparation',preparation_only=True,actual_saved_audit_selected=False,
        source_directory=source.as_posix(),source_sha256=files,original_root_files_sha256=original,
        unchanged_independent_decoder=dict(path=decoder.as_posix(),sha256=sha(decoder)),
        tests=dict(passed=40,failed=0,skipped=0,path=xml.as_posix(),sha256=sha(xml),python=sys.version,numpy=np.__version__),
        documentation_sha256=sha(BASE/'AUDIT_PREPARATION.md'),preparation_source_sha256=sha(Path(__file__)),
        original_task_array_evaluations=0,native_steps=0,model_calls=0,optimizer_updates=0,worker_processes=0,
        actual_request_created=False,original_root_sources_preserved=True,
        deltas=['Final uncaptured outer-cycle evidence and all available expert full291/history boundaries.',
                'Independent canonical job/result, admission, mailbox and worker timestamp reconstruction.',
                'Counter ordering, typed partial fault/capture evidence, reserved history records and fixed debt accounting.',
                'Immutable consumed-role/source/input/output/owner/process bindings and future gated saved-audit CLI.'])
    path=BASE/'source_preparation_audit_v1.json'
    with path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),source_files=len(files),tests=40)))
if __name__=='__main__':main()
