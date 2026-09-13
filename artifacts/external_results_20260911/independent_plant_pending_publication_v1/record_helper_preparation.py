"""Bind focused helper tests, actual PowerShell parse and unchanged source maps."""
import difflib,io,json,unittest
import xml.etree.ElementTree as ET
from prepare_concrete_packet import BASE,NEW,PRIOR,SOURCE,AUDIT,HELPERS,read,write,sha,SOURCE_PREP,ROOT_REVIEW,AUDIT_PREP,AUDIT_REVIEW
from prepare_clock_stage import texts

def main():
    suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_launch_helpers.py')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'helper_tests.log').open('x') as f:f.write(stream.getvalue())
    tree=ET.Element('testsuite',tests=str(result.testsRun),failures=str(len(result.failures)),errors=str(len(result.errors)),skipped=str(len(result.skipped)))
    with (BASE/'helper_tests.xml').open('xb') as f:f.write(ET.tostring(tree,encoding='utf-8',xml_declaration=True))
    if not result.wasSuccessful():raise ValueError('Helper tests failed; preserve evidence')
    parse=read(BASE/'template_parse.json')
    if parse.get('passed') is not True or parse.get('synthetic_only') is not True or parse.get('windows_path_normalization_passed') is not True:
        raise ValueError('Actual PowerShell template and path normalization checks required')
    for name,text in zip(('preview_run.ps1.txt','preview_durable.ps1.txt'),texts()):
        if (BASE/name).read_text()!=text or parse['source_sha256'][name]!=sha(BASE/name):raise ValueError('Wrong parsed template')
    old_names=('prepare_stage_preserved_template.py','stage_verdict.py','verify_completion.py')
    for name in old_names:
        if sha(BASE/name)!=sha(PRIOR/name):raise ValueError('Established helper changed: '+name)
    prep=read(BASE/'source_preparation.json');aprep=read(AUDIT/'source_preparation_v2.json')
    for name,digest in prep['source_sha256'].items():
        if sha(SOURCE/name)!=digest:raise ValueError('Producer source changed')
    for name,digest in aprep['source_sha256'].items():
        if sha(AUDIT/'source_draft_v2'/name)!=digest:raise ValueError('Auditor source changed')
    diff=''
    for name in ('prepare_concrete_packet.py','prepare_clock_stage.py'):
        diff+=''.join(difflib.unified_diff((PRIOR/name).read_text().splitlines(True),(BASE/name).read_text().splitlines(True),fromfile='timeout/'+name,tofile='pending/'+name))
    with (BASE/'helper_changes.diff').open('x') as f:f.write(diff)
    evidence=('helper_tests.log','helper_tests.xml','helper_changes.diff','preview_run.ps1.txt','preview_durable.ps1.txt','template_parse.json','check_templates.ps1')
    report=dict(passed=True,preparation_only=True,helper_sha256={name:sha(BASE/name) for name in HELPERS},
        evidence_sha256={name:sha(BASE/name) for name in evidence},tests=result.testsRun,failures=0,errors=0,skips=0,
        unchanged_helper_sha256={name:sha(BASE/name) for name in old_names},
        producer_source_sha256=prep['source_sha256'],auditor_source_sha256=aprep['source_sha256'],
        source_preparation_sha256=SOURCE_PREP,producer_review_sha256=ROOT_REVIEW,auditor_preparation_sha256=AUDIT_PREP,auditor_review_sha256=AUDIT_REVIEW,
        source_runtime_changes=0,task_arrays_loaded=0,native_steps=0,model_calls=0,optimizer_updates=0,spawned_workers=0,
        actual_dispatch=False,clearance_created=False)
    write(BASE/'helper_preparation.json',report)
    print(json.dumps(dict(helper_preparation_sha256=sha(BASE/'helper_preparation.json'),tests=result.testsRun)))

if __name__=='__main__':main()
