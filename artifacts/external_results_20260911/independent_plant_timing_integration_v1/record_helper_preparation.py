"""Freeze helper source/tests only. Actual auditor review is a future input."""
import argparse,difflib,io,json,unittest
import xml.etree.ElementTree as ET
from prepare_concrete_packet import (BASE,NEW,OLD,PRIOR,SOURCE,AUDIT,HELPERS,read,write,sha,
    SOURCE_PREP,ROOT_REVIEW,AUDIT_PREP,ORIGINAL_REQUEST,ORIGINAL_LAUNCH,check_review,antecedent_paths)
from prepare_clock_stage import texts

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    for name in ('clock_request.json','clock_process','run','stage_receipts','input_scope_derivation.json','helper_preparation.json'):
        if (BASE/name).exists():raise ValueError('Preserve prior artifact: '+name)
    if args.render_only:
        for name,text in zip(('preview_run.ps1.txt','preview_durable.ps1.txt'),texts()):
            with (BASE/name).open('x',encoding='utf-8') as f:f.write(text)
        return
    suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(BASE),pattern=name) for name in ('test_launch_helpers.py','test_sidecar_owner.py'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'helper_tests.log').open('x') as f:f.write(stream.getvalue())
    tree=ET.Element('testsuite',tests=str(result.testsRun),failures=str(len(result.failures)),errors=str(len(result.errors)),skipped=str(len(result.skipped)))
    with (BASE/'helper_tests.xml').open('xb') as f:f.write(ET.tostring(tree,encoding='utf-8',xml_declaration=True))
    if not result.wasSuccessful():raise ValueError('Preserve failed helper tests')
    parse=read(BASE/'template_parse.json')
    assert parse['passed'] is True and parse['synthetic_only'] is True and parse['windows_path_normalization_passed'] is True
    for name,text in zip(('preview_run.ps1.txt','preview_durable.ps1.txt'),texts()):
        assert (BASE/name).read_text()==text and parse['source_sha256'][name]==sha(BASE/name)
    old=BASE/'helper_original_v1'
    unchanged=('prepare_clock_stage.py','prepare_stage_preserved_template.py','check_templates.ps1')
    assert all(sha(BASE/name)==sha(PRIOR/name)==sha(old/name) for name in unchanged)
    prep_path=BASE/'source_preparation.json';review_path=NEW/'independent_timing_integration_root_review_v1/review.json'
    audit_prep=AUDIT/'source_preparation_v2.json';original_path=OLD/'clock_request.json';launch_path=OLD/'clock_process/launch_receipt.json'
    for path,digest in ((prep_path,SOURCE_PREP),(review_path,ROOT_REVIEW),(audit_prep,AUDIT_PREP),(original_path,ORIGINAL_REQUEST),(launch_path,ORIGINAL_LAUNCH)):
        assert sha(path)==digest,path
    prep=read(prep_path);aprep=read(audit_prep)
    check_review(prep,read(review_path),prep_path,SOURCE_PREP,32,'source_preparation_sha256')
    assert len(aprep['source_sha256'])==25
    for folder,entries in ((SOURCE,prep['source_sha256']),(AUDIT/'source_draft_v2',aprep['source_sha256'])):
        assert all(sha(folder/name)==digest for name,digest in entries.items())
    assert all(sha(SOURCE/name)==sha(PRIOR/'source_draft_v1'/name) for name in prep['unchanged_original_modules'])
    diff=''
    for name in HELPERS:
        prior=(old/name).read_text().splitlines(True) if (old/name).exists() else []
        diff+=''.join(difflib.unified_diff(prior,(BASE/name).read_text().splitlines(True),fromfile='preserved_pending_result/'+name,tofile='timing_helpers/'+name))
    with (BASE/'helper_derivation.diff').open('x',encoding='utf-8') as f:f.write(diff)
    # No fabricated review path/hash: actual packet CLI must supply and validate it.
    antecedents={}
    for path in antecedent_paths(prep_path,review_path,audit_prep,None,original_path,launch_path):
        if path is None or path==BASE/'helper_preparation.json':continue
        if not path.is_file():raise ValueError('Missing declared antecedent: '+str(path))
        antecedents[path.resolve().as_posix()]=sha(path)
    write(BASE/'antecedent_preflight.json',dict(passed=True,input_sha256=antecedents,
        all_existing_declared_paths_exist=True,actual_auditor_review_required_at_packet_freeze=True,
        actual_auditor_review_bound=False,task_arrays_loaded=0,clock_requests_created=0,native_steps=0,model_calls=0))
    evidence=('helper_tests.log','helper_tests.xml','helper_derivation.diff','preview_run.ps1.txt','preview_durable.ps1.txt',
        'template_parse.json','check_templates.ps1','antecedent_preflight.json','PACKET_SCOPE.md','OUTPUT_SCHEMA_DELTA.md','helper_derivation.json')
    value=dict(passed=True,preparation_only=True,helper_sha256={name:sha(BASE/name) for name in HELPERS},
        evidence_sha256={name:sha(BASE/name) for name in evidence},tests=result.testsRun,failures=0,errors=0,skips=0,
        unchanged_helper_sha256={name:sha(BASE/name) for name in unchanged},
        producer_source_sha256=prep['source_sha256'],auditor_source_sha256=aprep['source_sha256'],
        source_preparation_sha256=SOURCE_PREP,producer_review_sha256=ROOT_REVIEW,auditor_preparation_sha256=AUDIT_PREP,
        actual_auditor_review_required_at_packet_freeze=True,actual_auditor_review_bound=False,
        antecedent_path_count=len(antecedents),all_existing_declared_antecedents_exist=True,
        source_runtime_changes=0,task_arrays_loaded=0,native_steps=0,model_calls=0,optimizer_updates=0,spawned_workers=0,
        actual_requests_created=0,actual_launchers_created=0,actual_dispatch=False,clearance_created=False)
    write(BASE/'helper_preparation.json',value)
    print(json.dumps(dict(helper_preparation_sha256=sha(BASE/'helper_preparation.json'),tests=result.testsRun,
        producer_source_count=32,auditor_source_count=25,antecedents=len(antecedents))))

if __name__=='__main__':main()
