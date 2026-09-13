"""Freeze math/test/proposal preparation; never load task tensors or execute a fit."""
from pathlib import Path
import hashlib, json, ast, xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_draft_v1'
FINAL=BASE/'source_prepared_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
energy=json.loads((BASE/'energy_source.json').read_text())
for path,digest in energy['input_sha256'].items():assert sha(Path(path))==digest
suites=list(ET.parse(BASE/'tests_v1.xml').getroot().iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites)==16
assert all(int(s.get(k,'0'))==0 for s in suites for k in ('errors','failures','skipped'))
FINAL.mkdir(exist_ok=False)
for path in sorted(SOURCE.glob('*.py')):
    ast.parse(path.read_text())
    with (FINAL/path.name).open('xb') as stream:stream.write(path.read_bytes())
for name,digest in energy['original_source_sha256'].items():assert sha(FINAL/name)==digest
proposal=dict(preparation_only=True,root_selected=False,source_condition='causal',source_ordinary_step=68000,
    source_checkpoint_sha256='10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd',
    loss='N + original_coefficient * group_energy_balanced_F + P',coefficient=1.8188207859141674,
    group_order=energy['group_names'],group_weights=energy['group_weights'],group_energy_rule='Emean/Eg',
    proposed_additional_updates=3000,proposed_ordinary_final_step=71000,proposed_optimizer_steps=[3000,6000],
    proposed_learning_rate=[1e-6,1e-7],fresh_optimizer=False,saved_schedule_reused=True,sampler_draws=0,
    coefficient_recalibration=False,normalization_change=False,context_change=False,labels_change=False,
    original_metrics_preserved=True,weighted_metric_additional=True,numerical_tolerance_rad=1e-5,
    source_checkpoint_read=False,task_prediction_arrays_read=0,actual_task_model_calls=0,actual_task_gradient_calls=0,
    actual_optimizer_updates=0,actual_native_steps=0,causal_comparison_to_extra_epochs_claimed=False)
with (BASE/'proposal.json').open('x') as stream:json.dump(proposal,stream,indent=2);stream.write('\n')
subjects={name:dict(path=(BASE/name).as_posix(),sha256=sha(BASE/name)) for name in
          ('energy_source.json','PROPOSAL.md','proposal.json','tests_v1.xml','prepare_sources.py','record_preparation.py')}
result=dict(source_preparation_passed=True,preparation_only=True,root_selected=False,
    source_directory=FINAL.as_posix(),source_sha256={p.name:sha(p) for p in sorted(FINAL.glob('*.py'))},
    original_modules_byte_exact=energy['original_source_sha256'],subjects=subjects,synthetic_CPU_tests_passed=16,
    verified=['fixed teacher-only energies and exact weights','54cells dataset-phase-group order:9x6','constant-zero-response mean preserved tofloat64 roundoff',
              'underlying original metric and all54cells exact','analytic endpoint and repeated-center gradients','nominal value and gradient untouched','original dtype/shape rejection'],
    actual_task_model_calls=0,actual_task_gradient_calls=0,task_checkpoint_reads=0,task_prediction_array_reads=0,
    optimizer_updates=0,native_steps=0,training_request_created=False,controller_dispatch_performed=False)
with (BASE/'source_preparation.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),sources=len(result['source_sha256']),synthetic_tests=16,proposal_sha256=sha(BASE/'proposal.json'))))
