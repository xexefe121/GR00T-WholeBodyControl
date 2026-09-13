"""Capture bounded synthetic tests and immutable source identities; no real audit."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import time

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_draft_v1'
PYTHON='E:/codex_sonic_runtime/direct_target_gpu_20260911/venv/Scripts/python.exe'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    files=list(SOURCE.glob('*.py'))+[BASE/'prepare_audit_request.py']
    for path in files:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    copies={
        'audit_graph.py':BASE.parent/'direct_target_fp64_root_audit_v1/source_v2/audit_graph.py',
        'audit_restoration.py':BASE.parent/'direct_target_continuation_root_audit_v1/audit_restoration.py',
        'audit_math.py':BASE.parent/'direct_target_continuation_root_audit_v1/audit_math.py'}
    for name,path in copies.items():assert (SOURCE/name).read_bytes()==path.read_bytes()
    before={p.name:sha(p) for p in files};started=time.perf_counter()
    result=subprocess.run([PYTHON,'-m','unittest','-v','test_saved_math'],cwd=SOURCE,capture_output=True,text=True)
    (BASE/'synthetic_tests.stdout.log').write_text(result.stdout,encoding='utf-8');(BASE/'synthetic_tests.stderr.log').write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0 and 'Ran 12 tests' in result.stderr
    assert before=={p.name:sha(p) for p in files}
    report=dict(source_preparation_passed=True,source_directory=SOURCE.as_posix(),source_sha256={p.name:sha(p) for p in SOURCE.glob('*.py')},
        helper_sha256={str(BASE/'prepare_audit_request.py'):sha(BASE/'prepare_audit_request.py')},
        original_helpers={name:dict(path=path.as_posix(),sha256=sha(path),byte_identical=True) for name,path in copies.items()},
        synthetic_tests_passed=12,synthetic_exit_code=result.returncode,elapsed_seconds=time.perf_counter()-started,
        syntax_passed=True,stdout_sha256=sha(BASE/'synthetic_tests.stdout.log'),stderr_sha256=sha(BASE/'synthetic_tests.stderr.log'),
        source_reporter_sha256=sha(__file__),initial_test_failure_preserved='initial_synthetic_attempt.json',
        actual_fit_audit_executed=False,task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
        scope='Source and synthetic validation only. Root source review and completed actual fit are required before one saved-evidence audit.')
    with (BASE/'source_preparation.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print((BASE/'source_preparation.json').as_posix(),sha(BASE/'source_preparation.json'))
if __name__=='__main__':main()
