"""Freeze source-only integration; no task request, corpus or checkpoint reads."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
PRIOR=NEW/'direct_target_causal_width512_student_v1/source_prepared_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    draft=BASE/'source_draft_v1';snapshot=BASE/'source_prepared_v1'
    old={p.name:p for p in PRIOR.glob('*.py')}
    if len(old)!=28:raise ValueError('Original28 source files required')
    for name,path in old.items():
        if (draft/name).read_bytes()!=path.read_bytes():raise ValueError('Changed old source: '+name)
    copies={'warm512_restore.py':NEW/'direct_target_width251_warm_restore_v1/source_prepared_v1/warm512_restore.py',
            'input_schema.py':NEW/'direct_target_width251_collection_v1/source_prepared_v1/input_schema.py'}
    for name,path in copies.items():
        if (draft/name).read_bytes()!=path.read_bytes():raise ValueError('Changed qualified helper: '+name)
    for path in draft.glob('*.py'):ast.parse(path.read_text())
    suites=list(ET.parse(BASE/'synthetic_tests_v2.xml').getroot().iter('testsuite'))
    tests={name:sum(int(s.get(name,'0')) for s in suites) for name in ('tests','failures','errors','skipped')}
    if tests!={'tests':36,'failures':0,'errors':0,'skipped':0}:raise ValueError('Expected36 passing dedicated synthetic checks')
    snapshot.mkdir()
    for path in draft.glob('*.py'):shutil.copyfile(path,snapshot/path.name)
    source_map={p.name:sha(p) for p in sorted(snapshot.glob('*.py'))}
    references=[NEW/'direct_target_width251_warm_restore_v1/root_review.json',NEW/'direct_target_width251_collection_root_review_v1/review.json',
        NEW/'direct_target_width251_consistency_v1/root_review.json',NEW/'direct_target_width251_dagger_design_v1/DESIGN.md',
        NEW/'direct_target_width251_dagger_design_v1/CONSISTENCY_DIAGNOSTIC.md']
    report=dict(source_preparation_pass=True,prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_directory=snapshot.as_posix(),source_sha256=source_map,
        unchanged_original_sources={name:dict(path=p.as_posix(),sha256=sha(p)) for name,p in old.items()},
        unchanged_additional_sources={name:dict(path=p.as_posix(),sha256=sha(p)) for name,p in copies.items()},
        source_references={p.as_posix():sha(p) for p in references},
        tests=dict(path=(BASE/'synthetic_tests_v2.xml').as_posix(),sha256=sha(BASE/'synthetic_tests_v2.xml'),**tests),
        source_derivation_sha256=sha(BASE/'derive_integration.py'),main_derivation_sha256=sha(BASE/'main_derivation.patch'),
        output_schema_sha256=sha(BASE/'OUTPUT_SCHEMA.md'),preparation_source_sha256=sha(__file__),
        source_only=True,actual_fit_selected=False,actual_request_created=False,actual_task_data_reads=0,actual_checkpoint_reads=0,
        actual_model_calls=0,actual_gradient_calls=0,actual_optimizer_updates=0,native_steps=0,
        synthetic_gradient_tests=True,synthetic_fake_forward_counter_tests=True,
        legacy_fixed_protocol_tests_executed=False,root_source_review_required=True,
        unselected_protocol_fields=['updates','learning_rate_values','recovery_coefficient','ordinary_final_step','optimizer_final_step'],
        fixed_candidate_scope=dict(source_ordinary_step=81000,source_optimizer_step=16000,architecture=[1323,512,512,23],
            existing_schedule_rows=10000,schedule_wrapping=False,response_regeneration=False,
            nominal_cells=15,physical_cells=9,response_cells=54,recovery_cells=3,recovery_rows=1018,
            training_calls_per_update=4,training_rows_per_update=15704,diagnostic_calls_per_backend=1441,diagnostic_rows_per_backend=368588,
            initial_gate_corpora=3,FP64_parity_comparisons=12,parity_tolerance_rad=1e-5,expansion=False,normalization_refit=False))
    path=BASE/'source_preparation.json'
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),sources=len(source_map),old_exact=len(old),tests=tests)))
if __name__=='__main__':main()
