"""Prepare the selected recovery endpoint's release checks without loading a model."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'direct_target_causal_width512_evaluation_v1'
DEST = BASE / 'source_draft_v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)

assert not DEST.exists()
prep_path = OLD / 'source_preparation.json'
prep = json.loads(prep_path.read_text(encoding='utf-8-sig'))
for name, digest in prep['source_sha256'].items():
    source = OLD / 'source_draft_v1' / name
    assert sha(source) == digest, name
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(source.read_bytes())

path = DEST / 'evaluation_gate.py'
text = path.read_text()
text = replace_once(text, "binding['ordinary_final_step']==81000", "binding['ordinary_final_step']==91000")
text = replace_once(text, 'direct_absolute_target_1323_causal_width512', 'direct_absolute_target_1323_causal_width512_recovery')
text = replace_once(text, "'shared_manifest','context_alignment','energy_source')", "'shared_manifest','context_alignment','energy_source',\n           'recovery_rows','collection_report','collection_request','collection_qualification',\n           'collection_source_review','consistency_report','warm_restore_review')")
path.write_text(text, encoding='utf-8', newline='\n')

path = DEST / 'context_release.py'
text = path.read_text()
text = replace_once(text, 'Literal single width512 warm81000 release checks', 'Literal single width512 recovery91000 release checks')
text = replace_once(text, 'from balance_contract import GROUP_WEIGHTS, WEIGHT_RULE',
                    'from balance_contract import GROUP_WEIGHTS, WEIGHT_RULE\nfrom recovery_release import recovery_request_identity, validate_recovery_lineage')
text = replace_once(text, "'full_state_data_audit','full_state_data_owner','shared_manifest','context_alignment','energy_source')",
                    "'full_state_data_audit','full_state_data_owner','shared_manifest','context_alignment','energy_source',\n          'recovery_rows','collection_report','collection_request','collection_qualification',\n          'collection_source_review','consistency_report','warm_restore_review')")
text = replace_once(text, '395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d',
                    '825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e')
text = replace_once(text, 'condition=condition,ordinary_final_step=81000', 'condition=condition,ordinary_final_step=91000')
text = replace_once(text, 'additional_updates=10000,optimizer_step=16000', 'additional_updates=10000,optimizer_step=26000')
text = replace_once(text, 'ordinary_start_step=71000,optimizer_start_step=6000', 'ordinary_start_step=81000,optimizer_start_step=16000')
text = replace_once(text, 'architecture=[1323,512,512,23],hidden_width=512,expansion_seed=20260912,',
                    "architecture=[1323,512,512,23],hidden_width=512,expansion_performed=False,\n        recovery_rows=1018,recovery_phase_counts=[99,819,100],recovery_coefficient=0.2,\n        recovery_loss_columns=['recovery','weighted_recovery','combined_total'],\n        training_loss_columns=['nominal','balanced_full_state','physical','balanced_total','learning_rate','preclip_gradient_norm'],")
text = replace_once(text, "counts['training']==counter(30000,146860000)", "counts['training']==counter(40000,157040000)")
text = replace_once(text, 'counter(1437,367570)', 'counter(1441,368588)')
start = text.index('def training_request_identity(request):')
end = text.index('\n\ndef validate_release(', start)
text = text[:start] + 'def training_request_identity(request):\n    recovery_request_identity(request)\n' + text[end:]
text = replace_once(text, 'ordinary81000_width512_warm_balanced_context_same_weight_fp64_export',
                    'ordinary91000_width512_recovery_context_same_weight_fp64_export')
text = replace_once(text, '    training_request_identity(request)\n    assert request',
                    '    training_request_identity(request)\n    validate_recovery_lineage(binding, paths, request, fit, read=read, consumed_subjects=consumed_subjects)\n    assert request')
text = replace_once(text, "for script in ('evaluation_gate.py','context_release.py','causal_features.py','direct_runtime.py',",
                    "for script in ('evaluation_gate.py','context_release.py','recovery_release.py','causal_features.py','direct_runtime.py',")
path.write_text(text, encoding='utf-8', newline='\n')

path = DEST / 'test_context_release.py'
text = path.read_text()
for old, new in [
    ('condition=condition,ordinary_final_step=81000', 'condition=condition,ordinary_final_step=91000'),
    ('additional_updates=10000,optimizer_step=16000', 'additional_updates=10000,optimizer_step=26000'),
    ('ordinary_start_step=71000,optimizer_start_step=6000', 'ordinary_start_step=81000,optimizer_start_step=16000'),
    ('architecture=[1323,512,512,23],hidden_width=512,expansion_seed=20260912,',
     "architecture=[1323,512,512,23],hidden_width=512,expansion_performed=False,\n        recovery_rows=1018,recovery_phase_counts=[99,819,100],recovery_coefficient=0.2,\n        recovery_loss_columns=['recovery','weighted_recovery','combined_total'],\n        training_loss_columns=['nominal','balanced_full_state','physical','balanced_total','learning_rate','preclip_gradient_norm'],"),
    ('training=counter(30000,146860000)', 'training=counter(40000,157040000)'),
    ('counter(1437,367570)', 'counter(1441,368588)'),
    ("('expansion_seed',0)", "('expansion_performed',True),('recovery_rows',1017),('recovery_phase_counts',[100,819,99]),('recovery_coefficient',1.0)")]:
    text = replace_once(text, old, new)
path.write_text(text, encoding='utf-8', newline='\n')

# Retain the former width activation gate tests, replacing their obsolete request fixtures.
path = DEST / 'test_width_release.py'
text = path.read_text()
start = text.index('def request():')
end = text.index('\ndef binding():', start)
text = text[:start] + text[end:]
text = replace_once(text, 'ordinary_final_step=81000', 'ordinary_final_step=91000')
text = replace_once(text, 'direct_absolute_target_1323_causal_width512', 'direct_absolute_target_1323_causal_width512_recovery')
path.write_text(text, encoding='utf-8', newline='\n')

for name in ('recovery_release.py', 'test_recovery_release.py'):
    source = BASE / 'new_source' / name
    ast.parse(source.read_text(encoding='utf-8-sig'))
    with (DEST / name).open('xb') as stream:
        stream.write(source.read_bytes())

actual = {p.relative_to(DEST).as_posix(): sha(p) for p in sorted(DEST.rglob('*.py'))}
changed = [name for name, digest in prep['source_sha256'].items() if actual[name] != digest]
assert set(changed) == {'evaluation_gate.py', 'context_release.py', 'test_context_release.py', 'test_width_release.py'}
for path in DEST.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8-sig'))
diff = ''.join(''.join(difflib.unified_diff(
    (OLD / 'source_draft_v1' / name).read_text().splitlines(True),
    (DEST / name).read_text().splitlines(True), fromfile='preserved81000/' + name,
    tofile='recovery91000/' + name)) for name in changed)
with (BASE / 'source_changes.diff').open('x', encoding='utf-8') as stream:
    stream.write(diff)
result = dict(preparation_only=True, parent_preparation_sha256=sha(prep_path),
              original_source_sha256=prep['source_sha256'], source_sha256=actual,
              changed_sources=changed, unchanged_sources=len(prep['source_sha256'])-len(changed),
              added_sources=['recovery_release.py', 'test_recovery_release.py'],
              native_steps=0, task_array_reads=0, model_calls=0, tests_run=False,
              actual_binding_created=False, actual_evaluation_selected=False,
              writer_sha256=sha(Path(__file__)), source_changes_sha256=sha(BASE / 'source_changes.diff'))
with (BASE / 'runtime_derivation.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps({'sources':len(actual),'unchanged_sources':result['unchanged_sources'],
                  'runtime_derivation_sha256':sha(BASE / 'runtime_derivation.json')}))
