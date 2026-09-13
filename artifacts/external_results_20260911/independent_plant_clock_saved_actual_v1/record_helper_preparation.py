"""Bind helper-only synthetic checks and exact derivation; no clock audit execution."""
import difflib,io,unittest
import xml.etree.ElementTree as ET
from prepare_launch import *


def main():
    original=NEW/'direct_target_context_saved_semantics_review_v1'
    template=BASE/'preserved_saved_audit_template.ps1.txt'
    assert sha(template)==sha(original/'run_audit_durable.ps1')
    suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_launch.py')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'helper_tests.log').open('x') as f:f.write(stream.getvalue())
    tree=ET.Element('testsuite',tests=str(result.testsRun),failures=str(len(result.failures)),errors=str(len(result.errors)),skipped=str(len(result.skipped)))
    with (BASE/'helper_tests.xml').open('xb') as f:f.write(ET.tostring(tree,encoding='utf-8',xml_declaration=True))
    parse=read(BASE/'template_parse.json');assert parse['passed'] is True and parse['synthetic_only'] is True
    assert (BASE/'template_preview.ps1.txt').read_text()==render('a'*64)
    diff=''.join(difflib.unified_diff(template.read_text().splitlines(True),render('a'*64).splitlines(True),fromfile='preserved_causal/durable.ps1',tofile='clock_saved/synthetic_durable.ps1'))
    diff+=''.join(difflib.unified_diff((original/'verify_completion.py').read_text().splitlines(True),(BASE/'verify_completion.py').read_text().splitlines(True),fromfile='preserved_causal/owner.py',tofile='clock_saved/owner.py'))
    with (BASE/'helper_changes.diff').open('x') as f:f.write(diff)
    request=read(BASE/'request.json')
    for name,digest in request['source_sha256'].items():assert sha(SOURCE/name)==digest,name
    report=dict(passed=result.wasSuccessful(),helper_sha256={name:sha(BASE/name) for name in HELPERS},
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        original_template_sha256=sha(template),original_owner_sha256=sha(original/'verify_completion.py'),
        audit_source_sha256=request['source_sha256'],audit_source_review_sha256=SOURCE_REVIEW_SHA,
        request_sha256=sha(BASE/'request.json'),
        evidence_sha256={name:sha(BASE/name) for name in ('helper_tests.log','helper_tests.xml','helper_changes.diff','template_preview.ps1.txt','template_parse.json')},
        actual_audit_executed=False,clearance_created=False,dispatch_performed=False,model_calls=0,native_steps=0,task_arrays_loaded=0,
        changes=['Absolute frozen v3 audit source plus concrete request/output paths replace causal audit invocation.',
            'Saved integrity governs audit success while physical, command, timing and component failures remain visible.',
            'Original hidden process, handle, immutable records, exact concrete review, known exit and shared-delete streaming hashes retained.',
            'Owner validates clock report/comparison ledger and clock-owner subject; Windows/WSL input paths normalize with conflict rejection.'])
    write(BASE/'helper_preparation.json',report);assert result.wasSuccessful()
    print(json.dumps({'helper_preparation_sha256':sha(BASE/'helper_preparation.json'),'tests':result.testsRun,'helper_sha256':report['helper_sha256']}))


if __name__=='__main__':main()
