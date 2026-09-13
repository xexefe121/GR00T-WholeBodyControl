"""Record source derivation and synthetic checks; no actual saved audit."""
import ast
import io
import platform
import unittest
from saved_common import *

def main():
    paths=['saved_common.py','audit_saved.py','release_checks.py','fixed_maps.py','prepare_request.py','test_saved.py','record_preparation.py']
    sources={name:sha(BASE/name) for name in paths}
    original=NEW/'direct_target_fp64_fixed_map_v1/diagnose_fixed_map.py'
    functions=lambda path:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old,new=functions(original),functions(BASE/'fixed_maps.py')
    same={name:old[name]==new[name] for name in ('difference_function','committed_target','grouped')};assert all(same.values())
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_saved.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'synthetic_tests.log').open('x',encoding='utf-8') as f:f.write(stream.getvalue())
    report=dict(source_preparation_pass=result.wasSuccessful(),source_sha256=sources,tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),test_log_sha256=sha(BASE/'synthetic_tests.log'),
        preserved_original_semantics_source=dict(path=str(NEW/'direct_target_fp64_saved_semantics_review_v1/audit_saved.py'),sha256=sha(NEW/'direct_target_fp64_saved_semantics_review_v1/audit_saved.py')),
        preserved_original_map_source=dict(path=str(original),sha256=sha(original)),unchanged_map_function_AST=same,
        actual_evaluator_source_sha256={name:sha(SOURCE/name) for name in ('direct_features.py','direct_runtime.py','proposal_evidence.py','evaluate_direct_target_student.py','export_release_gate.py')},
        python=platform.python_version(),numpy=np.__version__,model_calls=0,BFM_calls=0,native_steps=0,optimizer_updates=0,
        actual_request_created=False,actual_audit_executed=False,
        changes=['Hardcoded292/42 replaced with actual issued main prefix and continuous terminal hold state/history handoff.',
            'Unified65000 release subjects replace old55000 failed-FP32/separate-FP64 release identities.',
            'Applied-target inverse applies only to learned phase; original raw BFM action remains startup/terminal feedback.',
            'Returned head output and failure capsule verified conditionally for actual phase; rejected commands never counted as issued.',
            'All actual learned rows receive same-clock immutable query250 map/group comparisons after semantic checks; no model, native or planning calls.'])
    write(BASE/'source_preparation.json',report)
    assert result.wasSuccessful()
    print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),tests=result.testsRun,sources=sources),indent=2))

if __name__=='__main__':main()
