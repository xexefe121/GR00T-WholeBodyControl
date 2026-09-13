"""Freeze one-line reduction correction and literal regression; no audit run."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;OLD=NEW/'direct_target_response_balanced_fit_independent_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
source=BASE/'source_prepared_v1';oldsource=OLD/'source_prepared_v1'
removed="            close('nominal_f32_mean',progress[:,0],wanted['nominal'],nominal=True)\n"
assert (source/'audit_saved_warm.py').read_text()==(oldsource/'audit_saved_warm.py').read_text().replace(removed,'')
unchanged={}
for name,d in read(OLD/'source_preparation.json')['source_sha256'].items():
    assert sha(oldsource/name)==d
    if name!='audit_saved_warm.py':assert sha(source/name)==d;unchanged[name]=d
helpers={n:sha(BASE/n) for n in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py')}
for n,d in helpers.items():assert sha(OLD/n)==d
tests=ET.parse(BASE/'tests_v1.xml').getroot()
assert len(tests.findall('.//testcase'))==50 and not any(tests.findall('.//'+k) for k in ('failure','error','skipped'))
refs=[OLD/'source_preparation.json',OLD/'results_v1/report.json',OLD/'owner_completion.json',
    NEW/'direct_target_response_fit_reduction_diagnosis_v1/report.json',NEW/'direct_target_response_fit_audit_root_review_v2/review.json',
    OLD/'saved_metadata_inspection.json',NEW/'direct_target_causal_response_balanced_student_v2/source_snapshot_v1/direct_objective.py',
    NEW/'direct_target_causal_response_balanced_student_v2/source_snapshot_v1/response_diagnostics.py']
assert not any((BASE/n).exists() for n in ('audit_request.json','launch_receipt.json','process_v1','results_v1'))
result=dict(source_preparation_passed=True,preparation_only=True,actual_audit_executed=False,
    source_directory=source.as_posix(),experiment=(NEW/'direct_target_causal_response_balanced_student_v2').as_posix(),
    source_sha256={p.name:sha(p) for p in sorted(source.glob('*.py'))},helper_sha256=helpers,
    unchanged_source_sha256=unchanged,removed_single_line=removed.strip(),all_other_production_source_unchanged=True,
    synthetic_tests_passed=50,new_tests=4,original_nominal_vs_float64_relative_tolerance=3e-7,model_parity_tolerance_rad=1e-5,
    reference_sha256={p.as_posix():sha(p) for p in refs},
    evidence_sha256={n:sha(BASE/n) for n in ('tests_v1.xml','source_delta.patch','derivation.json','prepare_source.py','record_preparation.py')},
    remaining_arithmetic_review=['Remaining five-backend metrics reuse qualified float64 full-state/physical differences and original nominal reduction contract.',
        'Added balanced metrics reproduce the producer 54-cell multiplication/order and six-group summaries with original coefficient; no new NumPy-float32 gate remains.',
        'All original and weighted cell/totals, shared context, output graph, parity/drift, counters, and process checks stay byte-identical.'],
    original_failed_audits_preserved=True,actual_numerical_audit_repeated=False,automatic_retry=False,
    task_array_loads=0,task_checkpoint_loads=0,task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0)
with (BASE/'source_preparation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(preparation_sha256=sha(BASE/'source_preparation.json'),sources=len(result['source_sha256']),helpers=len(helpers))))
