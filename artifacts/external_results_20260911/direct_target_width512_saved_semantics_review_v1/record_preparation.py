"""Freeze synthetic-only81000 semantics source, never read a task trace."""
import ast,difflib,io,platform,unittest
import xml.etree.ElementTree as ET
from saved_common import BASE,NEW,RUN,SOURCE,read,write,sha,np
from prepare_audit_stage import HELPERS,render,INVENTORY_SHA

def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def main():
    old=NEW/'direct_target_response_saved_semantics_review_v1';prior=read(old/'source_preparation.json')
    for name,digest in prior['source_sha256'].items():assert sha(old/name)==digest,name
    names=sorted(p.name for p in BASE.glob('*.py'));source={n:sha(BASE/n) for n in names}
    original={n:sha(old/n) for n in names if (old/n).exists()}
    identical={n:sha(BASE/n)==sha(old/n) for n in ('audit_saved.py','context_math.py','fixed_maps.py','test_saved.py','verify_completion.py')}
    assert all(identical.values()) and functions(BASE/'saved_common.py')==functions(old/'saved_common.py')
    assert functions(BASE/'prepare_audit_stage.py')==functions(old/'prepare_audit_stage.py')
    diff=''.join(''.join(difflib.unified_diff((old/n).read_text().splitlines(True) if (old/n).exists() else [],(BASE/n).read_text().splitlines(True),fromfile='preserved71000/'+n,tofile='width81000/'+n)) for n in names)
    with (BASE/'source_changes.diff').open('x',encoding='utf-8') as f:f.write(diff)
    suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_*.py');stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'synthetic_tests.log').open('x',encoding='utf-8') as f:f.write(stream.getvalue())
    root=ET.Element('testsuite',tests=str(result.testsRun),failures=str(len(result.failures)),errors=str(len(result.errors)),skipped=str(len(result.skipped)))
    for test,detail in result.failures+result.errors:ET.SubElement(root,'failure',name=test.id()).text=detail
    with (BASE/'synthetic_tests.xml').open('xb') as f:f.write(ET.tostring(root,encoding='utf-8',xml_declaration=True))
    assert result.wasSuccessful() and result.testsRun==61
    assert read(BASE/'stage_template_parse.json')['passed'] is True
    assert (BASE/'stage_template_preview.ps1.txt').read_text()==render('a'*64)
    assert sha(RUN/'runtime_inventory.json')==INVENTORY_SHA
    producer=RUN/'source_preparation.json';producer_map=read(producer)['source_sha256']
    for name,digest in producer_map.items():assert sha(SOURCE/name)==digest,name
    assert len(producer_map)==38
    assert not any((BASE/n).exists() for n in ('request.json','launch_receipt.json','launch_clearance.json','process_v1','results_v1','dispatch.json'))
    prep=dict(source_preparation_pass=True,source_sha256=source,original_source_sha256=original,
        byte_identical_files=identical,common_math_AST_identical=True,durable_helper_functions_AST_identical=True,
        tests_run=result.testsRun,failures=0,errors=0,skips=0,
        actual_producer_source_preparation={'path':producer.as_posix(),'sha256':sha(producer)},actual_producer_source_sha256=producer_map,
        runtime_inventory_sha256=INVENTORY_SHA,
        artifacts={n:sha(BASE/n) for n in ('source_changes.diff','synthetic_tests.log','synthetic_tests.xml','README.md','ROOT_REVIEW_GUIDE.md','stage_template_preview.ps1.txt','stage_template_parse.json','preserved_run_audit_template.ps1.txt')},
        python=platform.python_version(),numpy=np.__version__,ordinary_final_step=81000,architecture=[1323,512,512,23],context_condition='causal',features=1323,
        model_calls=0,ORT_calls=0,BFM_calls=0,native_steps=0,optimizer_updates=0,actual_task_arrays_loaded=0,
        actual_request_created=False,actual_audit_executed=False,preparation_only=True)
    write(BASE/'source_preparation.json',prep)
    write(BASE/'stage_source_preparation.json',dict(passed=True,synthetic_tests_passed=True,helper_sha256={n:sha(BASE/n) for n in HELPERS},
        audit_source_sha256=source,source_preparation_sha256=sha(BASE/'source_preparation.json'),tests_run=result.testsRun,
        original_template={'path':(old/'preserved_run_audit_template.ps1.txt').as_posix(),'sha256':sha(old/'preserved_run_audit_template.ps1.txt')},
        preserved_template_sha256=sha(BASE/'preserved_run_audit_template.ps1.txt'),preparation_only=True,actual_request_created=False,actual_dispatch_performed=False))
    print(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),stage_source_preparation_sha256=sha(BASE/'stage_source_preparation.json'),sources=len(source),tests=result.testsRun))
if __name__=='__main__':main()
