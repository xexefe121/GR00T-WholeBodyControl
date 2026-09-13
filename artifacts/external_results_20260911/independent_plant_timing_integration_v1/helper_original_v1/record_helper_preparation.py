"""Freeze source-only helper evidence; no actual request/launcher is written."""
import difflib,io,json,unittest
import xml.etree.ElementTree as ET
from prepare_concrete_packet import (BASE,NEW,OLD,PRIOR,SOURCE,AUDIT,HELPERS,read,write,sha,
    SOURCE_PREP,ROOT_REVIEW,AUDIT_PREP,AUDIT_REVIEW,ORIGINAL_REQUEST,ORIGINAL_LAUNCH,
    check_review,check_subject,antecedent_paths)
from prepare_clock_stage import texts

def main():
    for name in ('clock_request.json','clock_process','run','stage_receipts','input_scope_derivation.json','helper_preparation.json'):
        if (BASE/name).exists():raise ValueError('No preparation overwrite or actual execution artifact: '+name)
    suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_launch_helpers.py')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'helper_tests.log').open('x') as f:f.write(stream.getvalue())
    tree=ET.Element('testsuite',tests=str(result.testsRun),failures=str(len(result.failures)),errors=str(len(result.errors)),skipped=str(len(result.skipped)))
    with (BASE/'helper_tests.xml').open('xb') as f:f.write(ET.tostring(tree,encoding='utf-8',xml_declaration=True))
    if not result.wasSuccessful():raise ValueError('Helper tests failed; preserve evidence')
    parse=read(BASE/'template_parse.json')
    if parse.get('passed') is not True or parse.get('synthetic_only') is not True or parse.get('windows_path_normalization_passed') is not True:
        raise ValueError('Actual PowerShell template and normalization checks required')
    for name,text in zip(('preview_run.ps1.txt','preview_durable.ps1.txt'),texts()):
        if (BASE/name).read_text()!=text or parse['source_sha256'][name]!=sha(BASE/name):raise ValueError('Wrong parsed template')
    unchanged=('prepare_clock_stage.py','prepare_stage_preserved_template.py','stage_verdict.py','verify_completion.py','check_templates.ps1')
    for name in unchanged:
        if sha(BASE/name)!=sha(PRIOR/name):raise ValueError('Established helper changed: '+name)
    prep_path=BASE/'source_preparation.json';review_path=NEW/'independent_pending_result_source_review_v1/review.json'
    audit_prep=AUDIT/'source_preparation_v2.json';audit_review=NEW/'independent_pending_result_saved_audit_review_v1/review.json'
    original_path=OLD/'clock_request.json';launch_path=OLD/'clock_process/launch_receipt.json'
    for path,digest in ((prep_path,SOURCE_PREP),(review_path,ROOT_REVIEW),(audit_prep,AUDIT_PREP),
        (audit_review,AUDIT_REVIEW),(original_path,ORIGINAL_REQUEST),(launch_path,ORIGINAL_LAUNCH)):
        if sha(path)!=digest:raise ValueError('Reviewed subject changed: '+str(path))
    prep=read(prep_path);aprep=read(audit_prep);review=read(audit_review)
    check_review(prep,read(review_path),prep_path,SOURCE_PREP,24,'source_preparation_subject')
    check_review(aprep,review,audit_prep,AUDIT_PREP,20,'preparation_subject')
    check_subject(review.get('producer_preparation'),prep_path,SOURCE_PREP)
    check_subject(review.get('producer_source_review'),review_path,ROOT_REVIEW)
    for folder,entries in ((SOURCE,prep['source_sha256']),(AUDIT/'source_draft_v2',aprep['source_sha256'])):
        for name,digest in entries.items():
            if sha(folder/name)!=digest:raise ValueError('Reviewed source changed: '+name)
    diff=''.join(difflib.unified_diff((PRIOR/'prepare_concrete_packet.py').read_text().splitlines(True),
        (BASE/'prepare_concrete_packet.py').read_text().splitlines(True),fromfile='pending_publication/prepare_concrete_packet.py',tofile='pending_result/prepare_concrete_packet.py'))
    # The initial derivation receipt preceded final metadata preflight factoring.
    # Keep that receipt; publish the final exact diff as separate evidence.
    with (BASE/'helper_changes.diff').open('x') as f:f.write(diff)
    antecedents={}
    for path in antecedent_paths(prep_path,review_path,audit_prep,audit_review,original_path,launch_path):
        if path==BASE/'helper_preparation.json':continue # This receipt is being written below.
        if not path.is_file():raise ValueError('Missing declared antecedent: '+str(path))
        antecedents[path.resolve().as_posix()]=sha(path)
    for name in prep['unchanged_files']:
        if sha(SOURCE/name)!=sha(PRIOR/'source_draft_v1'/name):raise ValueError('Wrong original copy '+name)
    write(BASE/'antecedent_preflight.json',dict(passed=True,all_declared_paths_exist=True,
        excluded_self_receipt=(BASE/'helper_preparation.json').as_posix(),input_sha256=antecedents,
        task_arrays_loaded=0,clock_requests_created=0,native_steps=0,model_calls=0))
    evidence=('helper_tests.log','helper_tests.xml','helper_changes.diff','preview_run.ps1.txt','preview_durable.ps1.txt',
        'template_parse.json','check_templates.ps1','antecedent_preflight.json','PACKET_SCOPE.md','helper_derivation.json','helper_derivation.diff')
    report=dict(passed=True,preparation_only=True,helper_sha256={name:sha(BASE/name) for name in HELPERS},
        evidence_sha256={name:sha(BASE/name) for name in evidence},tests=result.testsRun,failures=0,errors=0,skips=0,
        unchanged_helper_sha256={name:sha(BASE/name) for name in unchanged},
        producer_source_sha256=prep['source_sha256'],auditor_source_sha256=aprep['source_sha256'],
        source_preparation_sha256=SOURCE_PREP,producer_review_sha256=ROOT_REVIEW,
        auditor_preparation_sha256=AUDIT_PREP,auditor_review_sha256=AUDIT_REVIEW,
        antecedent_path_count=len(antecedents),all_declared_antecedents_exist=True,
        source_runtime_changes=0,task_arrays_loaded=0,native_steps=0,model_calls=0,optimizer_updates=0,spawned_workers=0,
        actual_requests_created=0,actual_launchers_created=0,actual_dispatch=False,clearance_created=False)
    write(BASE/'helper_preparation.json',report)
    print(json.dumps(dict(helper_preparation_sha256=sha(BASE/'helper_preparation.json'),tests=result.testsRun,
        producer_source_count=len(prep['source_sha256']),auditor_source_count=len(aprep['source_sha256']),antecedents=len(antecedents))))

if __name__=='__main__':main()
