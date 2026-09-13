"""Additive helper provenance and synthetic tests; ten reviewed audit sources untouched."""
import difflib
import io
import unittest
import xml.etree.ElementTree as ET
from saved_common import BASE,sha,read,write
from prepare_audit_stage import OLD,REVIEW,REVIEW_SHA,HELPERS,render


def main():
    assert not (BASE/'request.json').exists() and not (BASE/'launch_receipt.json').exists()
    original=OLD/'run_audit_durable.ps1';template=BASE/'preserved_run_audit_template.ps1.txt'
    assert sha(original)==sha(template)
    prep=read(BASE/'source_preparation.json');review=read(REVIEW)
    assert sha(REVIEW)==REVIEW_SHA and review['source_sha256']==prep['source_sha256'] and review['source_review_pass'] is True
    for name,digest in prep['source_sha256'].items():assert sha(BASE/name)==digest,name
    helper_hashes={name:sha(BASE/name) for name in HELPERS}
    suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_audit_stage.py')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'stage_tests.log').open('x',encoding='utf-8') as output:output.write(stream.getvalue())
    tree=ET.Element('testsuite',tests=str(result.testsRun),failures=str(len(result.failures)),errors=str(len(result.errors)),skipped=str(len(result.skipped)))
    with (BASE/'stage_tests.xml').open('xb') as output:output.write(ET.tostring(tree,encoding='utf-8',xml_declaration=True))
    parse=read(BASE/'stage_template_parse.json');assert parse['passed'] is True and parse['synthetic_template_only'] is True
    assert (BASE/'stage_template_preview.ps1.txt').read_text()==render('a'*64)
    changes=''.join(difflib.unified_diff((OLD/'verify_completion.py').read_text().splitlines(True),
                                      (BASE/'verify_completion.py').read_text().splitlines(True),
                                      fromfile='preserved65000/verify_completion.py',tofile='causal68000/verify_completion.py'))
    changes+=''.join(difflib.unified_diff(template.read_text().splitlines(True),render('a'*64).splitlines(True),
                                       fromfile='preserved65000/durable.ps1',tofile='synthetic_causal68000/durable.ps1'))
    with (BASE/'stage_changes.diff').open('x',encoding='utf-8') as output:output.write(changes)
    report=dict(passed=result.wasSuccessful(),synthetic_tests_passed=result.wasSuccessful(),helper_sha256=helper_hashes,
        audit_source_sha256=prep['source_sha256'],audit_source_review={'path':REVIEW.as_posix(),'sha256':REVIEW_SHA},
        original_template={'path':original.as_posix(),'sha256':sha(original)},preserved_template_sha256=sha(template),
        original_owner_source_sha256=sha(OLD/'verify_completion.py'),tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        evidence_sha256={name:sha(BASE/name) for name in ('stage_tests.log','stage_tests.xml','stage_changes.diff',
            'stage_template_preview.ps1.txt','stage_template_parse.json')},
        preparation_only=True,actual_request_created=False,actual_launch_receipt_created=False,actual_clearance_created=False,
        dispatch_performed=False,model_calls=0,native_steps=0,task_arrays_loaded=0,
        changes=['Exact selected request and namespace replace old fixed65000 literals only after completed owner/native subjects exist.',
            'Concrete root request/launcher review required before lock/dispatch, bound before/after execution.',
            'Windows hidden WSL child, acquired handle, known exit, CreateNew records and .NET shared-delete streaming hashes retained.',
            'Future launch pins existing NumPy/SciPy package inventory and actual runtime bootstrap, not unused training corpora.',
            'Owner distinguishes successful saved evidence from preserved audit failure or unknown raw exit; no behavioral qualification.'])
    write(BASE/'stage_source_preparation.json',report)
    assert result.wasSuccessful()
    print({'stage_source_preparation_sha256':sha(BASE/'stage_source_preparation.json'),'tests':result.testsRun,'helper_sha256':helper_hashes})


if __name__=='__main__':main()
