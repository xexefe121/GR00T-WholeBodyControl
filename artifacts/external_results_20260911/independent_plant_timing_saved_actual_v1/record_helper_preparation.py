"""Freeze saved-audit helpers before any actual instrumented clock exists."""
import argparse,difflib,io,unittest
import xml.etree.ElementTree as ET
from prepare_launch import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    for name in ('request.json','launch_receipt.json','launch_clearance.json','process_v1','results_v1','dispatch.json','helper_preparation.json'):
        assert not (BASE/name).exists(),'Preparation only; preserve existing artifact'
    if args.render_only:
        with (BASE/'template_preview.ps1.txt').open('x',encoding='utf-8') as f:f.write(render('a'*64))
        return
    old=BASE/'helper_original_v1';template=BASE/'preserved_saved_audit_template.ps1.txt'
    assert sha(template)==sha(old/template.name)
    assert sha(BASE/'verify_completion.py')==sha(old/'verify_completion.py')
    suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_launch.py')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'helper_tests.log').open('x') as f:f.write(stream.getvalue())
    tree=ET.Element('testsuite',tests=str(result.testsRun),failures=str(len(result.failures)),errors=str(len(result.errors)),skipped=str(len(result.skipped)))
    with (BASE/'helper_tests.xml').open('xb') as f:f.write(ET.tostring(tree,encoding='utf-8',xml_declaration=True))
    assert result.wasSuccessful(),'Preserve failed synthetic evidence'
    parse=read(BASE/'template_parse.json');assert parse['passed'] is True and parse['synthetic_only'] is True
    assert parse['windows_path_normalization_passed'] is True
    assert (BASE/'template_preview.ps1.txt').read_text()==render('a'*64)
    assert parse['source_sha256']['template_preview.ps1.txt']==sha(BASE/'template_preview.ps1.txt')
    assert sha(SOURCE_PREPARATION)==SOURCE_PREPARATION_SHA
    source_map={p.name:sha(p) for p in SOURCE.glob('*.py')}
    assert len(source_map)==25 and read(SOURCE_PREPARATION)['source_sha256']==source_map
    diff=''
    for name in HELPERS:
        diff+=''.join(difflib.unified_diff((old/name).read_text().splitlines(True),(BASE/name).read_text().splitlines(True),fromfile='preserved_pending_result_saved/'+name,tofile='timing_saved_helpers/'+name))
    with (BASE/'helper_changes.diff').open('x',encoding='utf-8') as f:f.write(diff)
    evidence=('helper_tests.log','helper_tests.xml','helper_changes.diff','template_preview.ps1.txt','template_parse.json','check_template.ps1','derive_helpers.py','SCOPE.md')
    report=dict(passed=True,preparation_only=True,helper_sha256={name:sha(BASE/name) for name in HELPERS},
        tests=result.testsRun,failures=0,errors=0,skips=0,
        original_template_sha256=sha(template),original_owner_sha256=sha(old/'verify_completion.py'),
        audit_source_sha256=source_map,audit_preparation_sha256=SOURCE_PREPARATION_SHA,
        actual_source_review_required_from_future_request=True,actual_source_review_bound=False,
        evidence_sha256={name:sha(BASE/name) for name in evidence},
        actual_request_created=False,actual_audit_executed=False,clearance_created=False,dispatch_performed=False,
        model_calls=0,native_steps=0,optimizer_updates=0,task_arrays_loaded=0,
        changes=['Exact corrected25-file timing auditor replaces20-file pending-result auditor.',
            'Actual source review comes from completed-run request and must bind preparation/full map as explicit input.',
            'Original hidden durable template, mandatory receipt argument, raw exit and owner accounting remain unchanged.',
            'Evidence integrity remains separate from physical, timing, command and component qualification.'])
    write(BASE/'helper_preparation.json',report)
    print(json.dumps(dict(helper_preparation_sha256=sha(BASE/'helper_preparation.json'),tests=result.testsRun,actual_request_created=False)))

if __name__=='__main__':main()
