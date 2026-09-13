"""Source snapshot and JSON-schema proof only; no actual numeric files opened."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
DRAFT=BASE/'source_draft_v1'
SNAPSHOT=BASE/'source_prepared_v1'
sys.path.insert(0,str(DRAFT))
from admission import key,report_gate,PHYSICAL_KEYS
from test_consistency import fake_gate

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)

def main():
    if SNAPSHOT.exists():raise FileExistsError('Snapshot already exists; do not mutate or reprepare')
    fit=NEW/'direct_target_causal_width512_student_v1'
    subjects,reports,physical=fake_gate(BASE/'synthetic_future_collection')
    metadata_paths={
        'old_fit_report':fit/'fit/report.json',
        'old_fit_owner':fit/'owner_completion_verification.json',
        'old_fit_audit':NEW/'direct_target_width512_fit_independent_v1/results_v1/report.json',
        'old_training_request':fit/'training_request.json',
        'old_training_manifest':fit/'training_frozen_inputs.json',
        'old_shared_manifest':fit/'fit/shared/output_manifest.json',
        'old_context_alignment':fit/'fit/shared/context_alignment.json'}
    for role,path in metadata_paths.items():
        subjects[role]=dict(path=path.as_posix(),sha256=sha(path));reports[role]=read(path)
    request=reports['old_training_request'];frozen=reports['old_training_manifest'];shared=reports['old_shared_manifest']
    frozen_paths={key(path):digest for path,digest in frozen['input_sha256'].items()}
    for role,name in [('old_centers','centers'),('old_pico','pico'),('old_walk002','walk002'),('physical_manifest','physical_manifest'),('contract','contract')]:
        path=request['paths'][name].replace('\\','/');subjects[role]=dict(path=path,sha256=frozen_paths[key(path)])
    physical_manifest=Path(subjects['physical_manifest']['path']);reports['physical_manifest']=read(physical_manifest)
    if sha(physical_manifest)!=subjects['physical_manifest']['sha256']:raise ValueError('Metadata manifest changed')
    metadata_paths['physical_manifest']=physical_manifest
    for role,name in [('old_nominal_context','nominal_context.npy'),('old_physical_context','physical_context.npy'),('normalization','normalization.npz')]:
        subjects[role]=dict(path=(fit/'fit/shared'/name).as_posix(),sha256=shared['files'][name])
    for role in PHYSICAL_KEYS:
        spec=reports['physical_manifest']['arrays'][role]
        physical[role]=dict(path=(physical_manifest.parent/spec['path']).as_posix(),sha256=spec['sha256'])
    export=fit/'source_prepared_v1/width512_promoted.py'
    subjects['export_source']=dict(path=export.as_posix(),sha256=sha(export))
    reports['collection_report']['outputs']['normalization.npz']=subjects['normalization']['sha256']
    # Only old metadata is real. Future collection subjects/reports remain fake.
    report_gate(subjects,reports,physical)
    metadata_receipt=dict(passed=True,scope='actual old JSON schema and source identity; synthetic future collection',
        actual_numeric_files_read=0,actual_checkpoint_reads=0,task_model_calls=0,native_steps=0,
        metadata_read_sha256={p.as_posix():sha(p) for p in metadata_paths.values()},export_source_sha256=sha(export),
        future_collection_test_fixture=True,actual_collection_admitted=False)
    save(BASE/'metadata_schema_proof.json',metadata_receipt)
    copies={
        'direct_contract.py':fit/'source_prepared_v1/direct_contract.py',
        'input_schema.py':NEW/'direct_target_width251_collection_v1/source_prepared_v1/input_schema.py'}
    for name,path in copies.items():
        if (DRAFT/name).read_bytes()!=path.read_bytes():raise ValueError('Changed qualified copy: '+name)
    SNAPSHOT.mkdir()
    for path in sorted(DRAFT.glob('*.py')):shutil.copyfile(path,SNAPSHOT/path.name)
    tests=ET.parse(BASE/'synthetic_tests_v2.xml').getroot()
    suites=list(tests.iter('testsuite'))
    counts={name:sum(int(s.get(name,'0')) for s in suites) for name in ('tests','failures','errors','skipped')}
    if counts!={'tests':52,'failures':0,'errors':0,'skipped':0}:raise ValueError('Expected52 passing synthetic checks')
    receipt=dict(source_preparation_pass=True,prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_directory=SNAPSHOT.as_posix(),source_sha256={p.name:sha(p) for p in sorted(SNAPSHOT.glob('*.py'))},
        unchanged_sources={name:dict(path=p.as_posix(),sha256=sha(p)) for name,p in copies.items()},
        tests=dict(path=(BASE/'synthetic_tests_v2.xml').as_posix(),sha256=sha(BASE/'synthetic_tests_v2.xml'),**counts),
        metadata_schema_proof=dict(path=(BASE/'metadata_schema_proof.json').as_posix(),sha256=sha(BASE/'metadata_schema_proof.json')),
        source_references={p.as_posix():sha(p) for p in [export,fit/'source_prepared_v1/context_data.py',fit/'source_prepared_v1/response_data.py',
            fit/'source_prepared_v1/direct_data.py',NEW/'direct_target_width251_collection_v1/source_prepared_v1/collect_saved.py',
            NEW/'direct_target_width251_collection_v1/source_prepared_v1/qualification_gate.py',NEW/'direct_target_width251_collection_root_review_v1/review.json']},
        documentation_sha256={p.name:sha(p) for p in [BASE/'DESIGN.md',BASE/'OUTPUT_SCHEMA.md',BASE/'REVIEW_GUIDE.md']},
        preparation_source_sha256=sha(__file__),actual_request_created=False,actual_collection_admitted=False,
        actual_numeric_files_read=0,actual_checkpoint_reads=0,task_model_calls=0,gradient_calls=0,native_steps=0,optimizer_updates=0,
        root_source_review_required=True,root_concrete_review_required_before_actual_diagnosis=True,
        scope=dict(old_nominal=9904,old_physical=3054,new=1018,total=13976,alias_modes=6,proximity_queries=72,old_candidates_per_query=12958,candidate_block=256))
    save(BASE/'source_preparation.json',receipt)
    print(json.dumps(dict(path=(BASE/'source_preparation.json').as_posix(),sha256=sha(BASE/'source_preparation.json'),sources=len(receipt['source_sha256']),tests=counts)))

if __name__=='__main__':main()
