"""Record exact reviewed source/test identities; no task arrays or frameworks."""
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
STUDY=NEW/'direct_target_causal_response_balanced_student_v1'
PREP=STUDY/'source_preparation.json'
EXPECTED='1d9635f256508c34e4a8ffc7ba893da6c8fff755216fc878a94d8162b2135d2c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def subject(p):return dict(path=Path(p).as_posix(),sha256=sha(p))
assert sha(PREP)==EXPECTED
prep=read(PREP);source=Path(prep['source_directory'])
assert prep['source_preparation_passed'] is True and prep['preparation_only'] is True
assert len(prep['source_sha256'])==25
for name,digest in prep['source_sha256'].items():
    assert sha(source/name)==digest,(name,'frozen bytes')
    assert sha(STUDY/'source_draft_v1'/name)==digest,(name,'tested draft versus frozen')
    ast.parse((source/name).read_text(encoding='utf-8'))
old=NEW/'direct_target_causal_context_study_v2/source_prepared_v1'
for name,digest in prep['original_modules_byte_exact'].items():assert sha(old/name)==sha(source/name)==digest
math=NEW/'direct_target_response_balance_math_v1/source_prepared_v1'
for name,digest in prep['reviewed_math_modules_byte_exact'].items():assert sha(math/name)==sha(source/name)==digest
for value in prep['subjects'].values():assert sha(value['path'])==value['sha256']
def tests(p,expected):
    root=ET.parse(p).getroot();suites=[root] if root.tag=='testsuite' else root.findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites)==expected
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
tests(STUDY/'synthetic_tests_v3.xml',35)
tests(BASE/'tests_final_v2.xml',15)
proposal=read(STUDY/'training_request_proposal.json')
assert proposal['ordinary_start_step']==68000 and proposal['ordinary_final_step']==71000
assert proposal['optimizer_start_step']==3000 and proposal['optimizer_final_step']==6000
assert proposal['updates']==3000 and proposal['fresh_optimizer'] is False
assert proposal['learning_rate']==[1e-5,1e-6] and proposal['coefficient']==1.8188207859141674
assert proposal['root_selected'] is False
subjects=dict(source_preparation=subject(PREP),training_request_proposal=subject(STUDY/'training_request_proposal.json'),
    owner_synthetic_tests=subject(STUDY/'synthetic_tests_v3.xml'),independent_synthetic_tests=subject(BASE/'tests_final_v2.xml'),
    independent_test_source=subject(BASE/'test_independent_warm.py'),design=subject(STUDY/'DESIGN.md'),
    output_schema=subject(STUDY/'OUTPUT_SCHEMA.md'),energy_source=subject(NEW/'direct_target_response_balance_math_v1/energy_source.json'),
    balance_math_review=subject(NEW/'direct_target_response_balance_root_review_v1/review.json'),recorder=subject(__file__))
assert subjects['balance_math_review']['sha256']=='d999aba9c5181a15d922b14fd9e48df5efb03764172d2c5ef0c508e8a5108379'
report=dict(source_review_pass=True,warm_training_source_review_pass=True,preparation_only=True,
    prelaunch_review_pass=False,training_selected=False,execution_dispatched=False,
    source_directory=source.as_posix(),source_sha256=prep['source_sha256'],subjects=subjects,
    all_25_frozen_sources_match_tested_draft=True,original_modules_byte_exact=prep['original_modules_byte_exact'],
    reviewed_math_modules_byte_exact=prep['reviewed_math_modules_byte_exact'],
    independent_CPU_tests_passed=15,owner_CPU_tests_passed=35,findings=[],
    reviewed_scope=[
        'Exact six ordered learned1323 actor tensors and original full normalization; no expansion or context zeroing.',
        'Deep-copy warm AdamW load keeps independent expected source state; all six moments/scalarfloat32steps3000 and full parameter-group dictionary verified before LR override.',
        'CPU/CUDA/NumPy/Python RNG restored after constructor setup; no new calibration or schedule draws.',
        'Same3000 saved864-pair rows, ordinary68000to71000, AdamW3000to6000, inclusivecosine1e-5to1e-6, wd1e-5, clip10.',
        'Original nominal/physical losses and coefficient1.8188207859141674; reviewed positive six weights applied group-fastest across54 cells with live center gradients.',
        'All original F/cells/total retained alongside balancedF/cells/total, five complete diagnostics and fixed1e-5 preclamp gates; byte equality reported only.',
        'Exact9000 training calls/44058000rows; four Torch diagnostic passes5748calls/1470280rows plus ORT1437calls/367570rows; no tracing/BFM/native/calibration calls.',
        'Ordinary endpoint saved before export qualification; optimizer attempt/return/synchronization, six actual step counters, outputs and original/balanced loss prefixes retained on failure; no retry.'
    ],
    separate_root_scope=['Actual data/release gates, export qualification, final request and durable launcher review.'],
    verification_activity=dict(actual_task_model_calls=0,actual_task_gradient_calls=0,actual_task_checkpoint_loads=0,
        actual_task_prediction_arrays_loaded=0,native_steps=0,synthetic_CPU_test_runs=2,
        synthetic_optimizer_step_calls=2,synthetic_model_forward_calls=0,synthetic_backward_calls=0),
    review_limit='Source-only correctness review. Actual initial numerical gate and ordinary-final performance remain unmeasured; no fit or controller execution authorized here.')
with (BASE/'review.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(review=subject(BASE/'review.json'),source_review_pass=True,independent_tests=15,owner_tests=35)))
