"""Prepare retry-aware audit wrappers before any future completed clock exists."""
import difflib,io,unittest
import xml.etree.ElementTree as ET
from prepare_launch import *

def main():
    old=NEW/'independent_plant_pending_publication_saved_actual_v1'
    template=BASE/'preserved_saved_audit_template.ps1.txt'
    assert sha(template)==sha(old/template.name)
    assert sha(BASE/'verify_completion.py')==sha(old/'verify_completion.py')
    for name in ('request.json','launch_receipt.json','launch_clearance.json','process_v1','results_v1','dispatch.json'):
        assert not (BASE/name).exists(),'Preparation only; no actual packet/output allowed'
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
    assert sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA
    review=read(SOURCE_REVIEW);assert review['passed'] is True
    source_map={p.name:sha(p) for p in SOURCE.glob('*.py')}
    assert len(source_map)==20 and review['source_sha256']==source_map
    preparation=NEW/'independent_plant_pending_result_saved_audit_v1/source_preparation_v2.json'
    assert Path(review['preparation_subject']['path']).resolve()==preparation.resolve()
    assert review['preparation_subject']['sha256']==sha(preparation)=='09bf03da9bae5285ccc24cabd77a83718e06485dbc267a50d047deceea1f3d91'
    diff=''
    for name in HELPERS:
        diff+=''.join(difflib.unified_diff((old/name).read_text().splitlines(True),(BASE/name).read_text().splitlines(True),fromfile='prior_saved/'+name,tofile='pending_saved/'+name))
    with (BASE/'helper_changes.diff').open('x') as f:f.write(diff)
    evidence=('helper_tests.log','helper_tests.xml','helper_changes.diff','template_preview.ps1.txt','template_parse.json','check_template.ps1')
    report=dict(passed=True,preparation_only=True,helper_sha256={name:sha(BASE/name) for name in HELPERS},
        tests=result.testsRun,failures=0,errors=0,skips=0,
        original_template_sha256=sha(template),original_owner_sha256=sha(old/'verify_completion.py'),
        audit_source_sha256=source_map,audit_source_review_sha256=SOURCE_REVIEW_SHA,
        audit_source_review_path=SOURCE_REVIEW.as_posix(),
        evidence_sha256={name:sha(BASE/name) for name in evidence},
        actual_request_created=False,actual_audit_executed=False,clearance_created=False,dispatch_performed=False,
        model_calls=0,native_steps=0,optimizer_updates=0,task_arrays_loaded=0,
        changes=['Frozen retry-aware source_draft_v2 and its exact 20-file independent review replace old one-publication v3 source.',
            'Source preparation is independent of future completed-clock request, owner and output subjects.',
            'Original hidden durable wrapper, concrete review, known exit, streaming hashes and owner accounting remain unchanged.',
            'Evidence integrity remains separate from physical, timing, command and component qualification.'])
    write(BASE/'helper_preparation.json',report)
    print(json.dumps(dict(helper_preparation_sha256=sha(BASE/'helper_preparation.json'),tests=result.testsRun,actual_request_created=False)))

if __name__=='__main__':main()
