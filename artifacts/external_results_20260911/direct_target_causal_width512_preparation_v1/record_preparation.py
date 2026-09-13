"""Freeze source/synthetic evidence only; no checkpoint or model import."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')

xml=ET.parse(BASE/'synthetic_tests_v2.xml').getroot()
suites=[xml] if xml.tag=='testsuite' else list(xml)
assert sum(int(s.attrib['tests']) for s in suites)==27
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
names=['width512.py','test_width512.py','DESIGN.md']
snapshot=BASE/'source_prepared_v1';snapshot.mkdir(exist_ok=False)
for name in names:
    with (snapshot/name).open('xb') as f:f.write((BASE/name).read_bytes())
source_sha={name:sha(snapshot/name) for name in names}
assert source_sha=={name:sha(BASE/name) for name in names}
evidence=[BASE/'synthetic_tests_v2.xml',BASE/'synthetic_tests_v2.log',
          BASE/'preserved_synthetic_attempt_v1/synthetic_tests_v1.xml',
          BASE/'preserved_synthetic_attempt_v1/synthetic_tests_v1.log',
          NEW/'direct_target_balanced_convergence_review_v1/report.json',
          NEW/'direct_target_balanced_convergence_review_v1/evidence_summary.md',
          NEW/'direct_target_causal_response_balanced_student_v2/source_prepared_v1/context_model.py']
proposal=dict(kind='unselected_width512_capacity_engineering_proposal',actual_fit_selected=False,
              future_source_checkpoint_sha256='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d',
              source_checkpoint_not_loaded_or_rehashed=True,architecture=[1323,512,512,23],
              old_architecture=[1323,256,256,23],source_step=71000,final_step=81000,updates=10000,
              optimizer_start_step=6000,optimizer_final_step=16000,fresh_optimizer=False,
              group_count=1,parameter_state_count=6,new_moments='zero',old_moments='exact_old_blocks',
              shared_step_bias_correction=True,expansion_seed=20260912,
              local_expansion_rng_separate_from_source_global_rng=True,
              learning_rate=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],cosine_updates=9750,cosine_inclusive=[1e-5,1e-6]),
              weight_decay=1e-5,gradient_clip=10,coefficient=1.8188207859141674,
              objective='unchanged causal nominal + coefficient*teacher-energy-balanced54-cell full-state + physical',
              qualified_schedule_reuse_required=True,training_rows=146860000,training_forwards=30000,
              calibration_calls=0,planned_Torch_diagnostic_rows=1470280,planned_Torch_diagnostic_calls=5748,
              planned_ORT_diagnostic_rows=367570,planned_ORT_diagnostic_calls=1437,
              public_dtype='float32',export_internal_dtype='float64',parity_tolerance_rad=1e-5,
              learned_BFM_calls=0,native_main_controls=1569,conditional_hold_controls=250,
              end_to_end_including_features_latency_requirement_ms=20,
              actual_task_calls_permitted_by_this_proposal=0)
save(BASE/'proposal.json',proposal)
result=dict(source_preparation_passed=True,preparation_only=True,source_directory=snapshot.as_posix(),
            source_sha256=source_sha,proposal_sha256=sha(BASE/'proposal.json'),
            synthetic_CPU_tests=27,tests_failed=0,tests_skipped=0,
            preserved_first_test_attempt=dict(tests=27,passed=26,failed=1,
                reason='Synthetic AdamW fixture omitted PyTorch2.10 decoupled_weight_decay=True; fixture and explicit validation corrected, original attempt retained.'),
            tested_properties=[
                'Eight CPU batch sizes1,52,176,238,256,1728,3054,9904 preserve whole original network bytes.',
                'Old learned blocks and all six moments/steps/groups exact; added moment entries zero; global RNG unchanged.',
                'Two synthetic gradient updates open zero outgoing paths and then change new incoming weights.',
                'Nonzero split/dense512 output and all-six-gradient comparisons pass declared float32 arithmetic tolerance.',
                'Signed-zero forward identity and first/second derivatives of zero-preserving addition pass.',
                'Warm synthetic AdamW load and zero-gradient step retain exact old blocks and advance six counters6000→6001.',
                'Corrupted metadata, normalization, moments and steps reject; fixed schedule endpoints/counts pass.'],
            limitations=[
                'CPU synthetic proof only; actual CUDA checkpoint/parity/timing remains unqualified.',
                'Shared step6000 on new zero moments differs from a fresh optimizer trajectory.',
                'Width and ramp proposal is an engineering change, not unique proof of capacity causation.',
                'No manual exporter, actual fit runner, actual request, witness or canonical package created.'],
            input_sha256={p.as_posix():sha(p) for p in evidence},writer_sha256=sha(__file__),
            actual_checkpoint_loads=0,actual_task_model_calls=0,actual_gradient_calls=0,
            actual_optimizer_updates=0,native_steps=0,actual_training_selected=False)
save(BASE/'source_preparation.json',result)
print(json.dumps(dict(preparation_sha256=sha(BASE/'source_preparation.json'),
                      proposal_sha256=sha(BASE/'proposal.json'),source_sha256=source_sha)))
