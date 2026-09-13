"""Preserved v1, final path-normalization fix and actual PowerShell stub check."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
from prepare_bound_launcher import BASE,sha,read,run_text,durable_text,write_new,write_json
from record_helper_preparation import HELPERS


def main():
    prep=read(BASE/'source_preparation.json');inventory=read(BASE/'runtime_inventory.json')
    sources={p.relative_to(BASE/'source_draft_v1').as_posix():sha(p) for p in (BASE/'source_draft_v1').rglob('*.py')}
    assert sources==prep['source_sha256']==inventory['source_sha256']
    old=BASE/'launch_helpers_preserved_v1/launch_helper_preparation.json'
    assert sha(old)=='d46933f64315157329d5fb64206ccd732e737a49e972f168cc591f65ecb50e39'
    helpers={name:sha(BASE/name) for name in HELPERS}
    suites=ET.parse(BASE/'launch_helper_tests_v2.xml').getroot()
    assert len(suites.findall('.//testcase'))==27 and not any(suites.findall('.//'+name) for name in ('failure','error','skipped'))
    normalization=read(BASE/'powershell_path_normalization_v2.json')
    assert normalization['passed'] is True and normalization['actual']==normalization['expected']
    assert normalization['expression']=='Replace([char]92,[char]47)'
    for name in ('release_reviews.json','witness_binding.json','evaluation_binding.json','witness_process','evaluation_process','head_witness','nominal'):
        assert not (BASE/name).exists(),name
    directory=BASE/'synthetic_launcher_templates_v2';directory.mkdir(exist_ok=False)
    for mode in ('witness','evaluation'):
        folder=directory/(mode+'_fake_process')
        write_new(directory/(mode+'_run.ps1'),run_text(BASE,folder,mode,'a'*64))
        write_new(directory/(mode+'_durable.ps1'),durable_text(folder))
    result={'preparation_passed':True,'preparation_only':True,'helper_sha256':helpers,'source_sha256':sources,
        'source_preparation_sha256':sha(BASE/'source_preparation.json'),'preserved_v1_preparation_sha256':sha(old),
        'runtime_inventory_sha256':sha(BASE/'runtime_inventory.json'),'runtime_files':inventory['total_files'],
        'baseline_full291_fixture_byteexact':inventory['baseline_full291_fixture_byteexact'],'synthetic_tests_passed':27,
        'synthetic_tests_sha256':sha(BASE/'launch_helper_tests_v2.xml'),
        'powershell_normalization_sha256':sha(BASE/'powershell_path_normalization_v2.json'),
        'synthetic_template_sha256':{p.name:sha(p) for p in directory.glob('*.ps1')},
        'actual_bindings_created':False,'actual_selection_record_created':False,'model_calls':0,'native_steps':0,
        'optimizer_updates':0,'recursive_training_hashes':False,'separate_actual_final_review_required':True,
        'helper_review_contract':'passed true and helper_sha256 exactly this eight-file map',
        'final_stage_review_contract':'passed true plus literal binding_subject and launch_receipt_subject objects, each actual path and SHA',
        'record_source_sha256':sha(__file__)}
    write_json(BASE/'launch_helper_preparation_v2.json',result)
    print(json.dumps({'preparation_sha256':sha(BASE/'launch_helper_preparation_v2.json'),'helpers':8,'runtime_sources_unchanged':32,'synthetic_tests':27}))


if __name__=='__main__':main()
