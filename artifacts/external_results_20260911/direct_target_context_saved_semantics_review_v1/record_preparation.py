"""Freeze only sources and synthetic evidence. No actual request or task audit."""
import ast
import difflib
import io
import platform
import unittest
import xml.etree.ElementTree as ET
from saved_common import *


def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def flatten(suite):
    for entry in suite:
        if isinstance(entry,unittest.TestSuite):yield from flatten(entry)
        else:yield entry


def main():
    old=NEW/'direct_target_full_state_saved_semantics_review_v1'
    names=['saved_common.py','audit_saved.py','release_checks.py','fixed_maps.py','prepare_request.py',
           'context_math.py','test_saved.py','test_context_math.py','derive_sources.py','record_preparation.py']
    sources={name:sha(BASE/name) for name in names}
    originals={name:sha(old/name) for name in names if (old/name).exists()}
    unchanged={}
    for name,keep in [('saved_common.py',None),('audit_saved.py',['check_inference']),('fixed_maps.py',None)]:
        before,after=functions(old/name),functions(BASE/name)
        selected=before if keep is None else keep
        for function in selected:
            identity=name+':'+function
            unchanged[identity]=before[function]==after[function]
    assert all(unchanged.values())
    assert sha(old/'fixed_maps.py')==sha(BASE/'fixed_maps.py')
    changes=''.join(''.join(difflib.unified_diff((old/name).read_text().splitlines(True) if (old/name).exists() else [],
        (BASE/name).read_text().splitlines(True),fromfile='preserved65000/'+name,tofile='causal68000/'+name)) for name in names)
    with (BASE/'source_changes.diff').open('x',encoding='utf-8') as stream:stream.write(changes)
    suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_*.py');tests=list(flatten(suite))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'synthetic_tests.log').open('x',encoding='utf-8') as output:output.write(stream.getvalue())
    tree=ET.Element('testsuite',tests=str(result.testsRun),failures=str(len(result.failures)),errors=str(len(result.errors)),skipped=str(len(result.skipped)))
    failures={test.id():message for test,message in result.failures};errors={test.id():message for test,message in result.errors}
    for test in tests:
        case=ET.SubElement(tree,'testcase',name=test.id())
        for tag,items in [('failure',failures),('error',errors)]:
            if test.id() in items:ET.SubElement(case,tag).text=items[test.id()]
    with (BASE/'synthetic_tests.xml').open('xb') as output:output.write(ET.tostring(tree,encoding='utf-8',xml_declaration=True))
    producer_preparation=RUN/'source_preparation.json';producer_map=read(producer_preparation)['source_sha256']
    for name,digest in producer_map.items():assert sha(SOURCE/name)==digest,name
    report=dict(source_preparation_pass=result.wasSuccessful(),source_sha256=sources,original_source_sha256=originals,
        unchanged_function_AST=unchanged,fixed_map_file_byte_identical=True,
        actual_producer_source_preparation={'path':producer_preparation.as_posix(),'sha256':sha(producer_preparation)},
        actual_producer_source_sha256=producer_map,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        artifacts={name:sha(BASE/name) for name in ['source_changes.diff','synthetic_tests.log','synthetic_tests.xml','README.md']},
        python=platform.python_version(),numpy=np.__version__,model_calls=0,BFM_calls=0,native_steps=0,optimizer_updates=0,
        actual_task_arrays_loaded=0,actual_goal_or_map_evaluations=0,actual_request_created=False,actual_audit_executed=False,
        context_condition='causal',public_features=1323,original_requested_controls=1569,conditional_continuous_hold_controls=250,
        changes=['Current1000 feature reconstruction remains the original pure saved-state/received-goal implementation.',
            'Independent incoming prior23 and sorted named pre-update H300 form the additional323 public inputs; no context mean substitution.',
            'Original state52, applied-target inverse, next H, BFM raw feedback, source clocks, repeated native time and continuous hold checks preserved.',
            'Causal1323 baseline/query250/witness checks replace reduced1000-only input comparison.',
            'Paired68000 causal18 subjects replace unified65000 release13 subjects; original process/native evidence accounting retained.',
            'Rejected head-input/staged history checks added. An issued final command may have zero returned steps; earlier commands still require ten.',
            'Unchanged saved58 map math runs only for existing unique matching controls; missing controls reported, ambiguous mappings fail.'])
    write(BASE/'source_preparation.json',report)
    assert result.wasSuccessful()
    assert not (BASE/'request.json').exists() and not (BASE/'results_v1').exists()
    print(json.dumps({'source_preparation_sha256':sha(BASE/'source_preparation.json'),'tests':result.testsRun,
                      'source_sha256':sources,'actual_audit_executed':False},indent=2))


if __name__=='__main__':main()