"""Freeze source and completed synthetic results; never load task arrays."""
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def subject(path):return dict(path=path.as_posix(),sha256=sha(path))
copied=json.loads((BASE/'copied_sources.json').read_text())
for name,item in copied.items():assert sha(BASE/'source_draft_v1'/name)==sha(Path(item['path']))==item['sha256']
suites=ET.parse(BASE/'synthetic_tests_v2.xml').getroot().findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites)==28
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
prepared=BASE/'source_prepared_v1';prepared.mkdir(exist_ok=False)
source={}
for path in sorted((BASE/'source_draft_v1').glob('*.py')):
    shutil.copyfile(path,prepared/path.name);source[path.name]=sha(path);assert source[path.name]==sha(prepared/path.name)
assert len(source)==9
references=[NEW/'direct_target_width251_followup_design_v1/DESIGN.md',
    NEW/'bfm250_expert_root_qualification_v1/source.py',
    NEW/'direct_target_width512_saved_semantics_review_v1/fixed_maps.py',
    NEW/'direct_target_width512_expert_recovery_v2/source_snapshot_v1/run_actual_student_oracle.py',
    NEW/'direct_target_width512_expert_recovery_v2/source_snapshot_v1/run_width251_actual_oracle.py',
    NEW/'direct_target_width512_expert_recovery_v2/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py',
    NEW/'direct_target_width512_expert_recovery_v2/verify_recovery_completed.py']
report=dict(passed=True,source_preparation_only=True,actual_collection_selected=False,actual_collection_executed=False,
    model_fitting_authorized=False,source_directory=prepared.as_posix(),source_sha256=source,
    unchanged_pure_modules=copied,source_count=9,unchanged_module_count=4,
    tests=dict(passed=True,count=28,receipt=subject(BASE/'synthetic_tests_v2.xml'),
               scope='Synthetic arrays/receipts and source AST identities only; no task data arrays or models.'),
    source_references={p.as_posix():sha(p) for p in references},
    design=subject(BASE/'DESIGN.md'),output_schema=subject(BASE/'OUTPUT_SCHEMA.md'),
    required_actual_scope=dict(control_start=251,control_stop_exclusive=1269,rows=1018,phase_rows=[99,819,100],
        fresh_student_state_queries=1,connected_expert_rows_after_first=1017,committed_plans=204,
        required_main_controls=1569,required_continuous_hold_controls=250,required_independent_reports=4),
    task_arrays_opened=0,task_model_calls=0,bfm_actor_calls=0,bfm_backward_calls=0,native_steps=0,replans=0,optimizer_updates=0,
    limitations=['Recovery must first complete and pass four independent physics/intent reports plus owner accounting.',
        'Only source preparation is complete. No actual request, labels, normalization fit or training selection exists.',
        'Later connected expert states and committed gain maps are not independently queried failed student states.'],
    preparation_sources={p.name:sha(p) for p in (BASE/'prepare_source.py',Path(__file__))})
with (BASE/'source_preparation.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(receipt=subject(BASE/'source_preparation.json'),sources=9,tests=28,actual_collection=False)))
