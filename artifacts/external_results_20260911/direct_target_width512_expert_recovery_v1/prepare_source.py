"""Freeze source/proposed protocol only. Never loads task arrays or native code."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SOURCE=BASE/'source_draft_v1'
PREPARED=BASE/'source_prepared_v1'
sys.path.insert(0,str(SOURCE))
from recovery_contract import protocol,SOURCE_TRACE_SHA,SEMANTICS_SHA,SEMANTICS_OWNER_SHA

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def subject(p):return dict(path=str(p).replace('\\','/'),sha256=sha(p))
def write(p,obj):
    with p.open('x',newline='\n') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def main():
    tests=BASE/'synthetic_tests_final.xml'
    suites=ET.parse(tests).getroot().findall('testsuite')
    total=sum(int(s.attrib['tests']) for s in suites)
    assert total>=22 and all(int(s.attrib['failures'])==int(s.attrib['errors'])==0 for s in suites)
    old=NEW/'bfm_entry250_actual_oracle_v1'
    old_receipt=json.loads((old/'frozen_inputs.json').read_text())
    old_sources={k:v for k,v in old_receipt['source_sha256'].items() if k!='run_bfm250_actual_oracle.py'}
    sources={p.relative_to(SOURCE).as_posix():sha(p) for p in sorted(SOURCE.rglob('*.py'))}
    assert len(sources)==20 and len(old_sources)==14
    for name,digest in old_sources.items():assert sources[name]==digest
    for p in SOURCE.rglob('*.py'):ast.parse(p.read_text())
    PREPARED.mkdir(exist_ok=False)
    for name,digest in sources.items():
        p=PREPARED/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((SOURCE/name).read_bytes());assert sha(p)==digest
    old_adapter=old/'source_snapshot_v1/run_bfm250_actual_oracle.py'
    (BASE/'adapter_changes.diff').write_text(''.join(difflib.unified_diff(old_adapter.read_text().splitlines(True),
        (SOURCE/'run_width251_actual_oracle.py').read_text().splitlines(True),fromfile='qualified_BFM250_adapter',tofile='actual_width251_adapter')))
    subjects={
        'trace':dict(path=str(NEW/'direct_target_causal_width512_evaluation_v1/nominal/trace.npz').replace('\\','/'),sha256=SOURCE_TRACE_SHA),
        'semantics':subject(NEW/'direct_target_width512_saved_semantics_review_v1/results_v1/report.json'),
        'semantics_owner':subject(NEW/'direct_target_width512_saved_semantics_review_v1/owner_completion.json'),
        'canonical_owner':subject(NEW/'direct_target_causal_width512_evaluation_v1/evaluation_completion_verification.json'),
        'native_replay':subject(NEW/'direct_target_width512_independent_physics_v1/report.json'),
        'intent':subject(NEW/'direct_target_width512_independent_intent_v1/report.json'),
        'old_oracle_frozen':subject(old/'frozen_inputs.json'),
        'fallback_review':subject(NEW/'direct_target_width512_fallback_review_v1/report.json'),
        'contract':subject(Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')),
    }
    assert subjects['semantics']['sha256']==SEMANTICS_SHA and subjects['semantics_owner']['sha256']==SEMANTICS_OWNER_SHA
    # Trace identity is carried from audited receipts here; actual preparation
    # and launch must rehash it. No task NPZ is opened during source preparation.
    proposal=dict(kind='fixed_width81000_pre251_actual_expert_recovery',root_selected=False,
        source_only=True,protocol=protocol(),subjects=subjects,
        required_future_subjects={'selected_snapshot':'inputs/precontrol251.npz','selected_prefix':'inputs/actual_prefix251.npz',
                                  'input_selection':'inputs/selection_receipt.json'},
        actual_input_preparation_run=False,actual_recovery_run=False,
        execution='source_snapshot_v1/run_width251_actual_oracle.py; no rollout/checkpoint arguments',
        environment=dict(python='/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
                         PYTHONPATH='fixed source_snapshot_v1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        reviewer_and_root_concrete_clearance_required=True)
    write(BASE/'execution_request.proposal.json',proposal)
    write(BASE/'input_preparation_request.proposal.json',dict(root_selected_saved_input_preparation=False,
        subjects={k:subjects[k] for k in ('trace','semantics','semantics_owner','contract')},output=str(BASE/'inputs').replace('\\','/')))
    # These are lineage pins, not a fabricated current inventory qualification.
    catalog={k:v for k,v in old_receipt['input_sha256'].items()
             if '/bfm_entry250_actual_oracle_v1/inputs/' not in k and '/original_bfm_entry250' not in k}
    write(BASE/'inherited_input_catalog.json',dict(source=subjects['old_oracle_frozen'],input_sha256=catalog,
        current_rehash_performed=False,actual_freeze_must_add_new_selected_inputs_and_reviews=True))
    write(BASE/'source_preparation.json',dict(source_preparation_pass=True,source_only=True,
        source_directory=str(PREPARED).replace('\\','/'),source_sha256=sources,unchanged_qualified_source_sha256=old_sources,
        subjects=subjects,protocol=protocol(),synthetic_tests=dict(passed=True,tests=total,receipt=subject(tests),
            task_array_loads=0,task_model_calls=0,native_steps=0,optimizer_calls=0),
        artifacts={p.name:subject(p) for p in [BASE/'DESIGN.md',BASE/'OUTPUT_SCHEMA.md',BASE/'derive_source.py',
            BASE/'prepare_source.py',BASE/'adapter_changes.diff',BASE/'execution_request.proposal.json',
            BASE/'input_preparation_request.proposal.json',BASE/'inherited_input_catalog.json']},
        first_collection_failure_preserved='synthetic_tests_v1.xml; invalid pytest root traversed E:/WpSystem; no tests collected',
        actual_saved_input_preparation=False,actual_recovery_selected=False))
    print(json.dumps(dict(source_preparation=subject(BASE/'source_preparation.json'),sources=len(sources),tests=total)))

if __name__=='__main__':main()
