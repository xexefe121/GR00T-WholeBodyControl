"""Prepare81000 checker only; concrete witness subjects must be supplied later."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'direct_target_response_canonical_concrete_review_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parent_review = json.loads((OLD / 'review.json').read_text())
assert parent_review['passed'] and sha(OLD / 'review_concrete.py') == parent_review['writer_sha256']
original = (OLD / 'review_concrete.py').read_text()
source = original
changes = []


def change(old, new, count=1):
    global source
    assert source.count(old) == count, (old, source.count(old), count)
    source = source.replace(old, new)
    changes.append(dict(before=old, after=new, occurrence_count=count))


change('causal71000', 'causal81000 width512')
change('direct_target_causal_response_evaluation_v1', 'direct_target_causal_width512_evaluation_v1')
change("parser.add_argument('--witness-sha256', required=True)",
       "parser.add_argument('--witness-sha256', required=True)\n"
       "parser.add_argument('--witness-binding-sha256', required=True)\n"
       "parser.add_argument('--witness-launch-sha256', required=True)\n"
       "parser.add_argument('--witness-review-sha256', required=True)")
change('assert args.binding_pins > 5226 and args.launch_pins == args.binding_pins + 5',
       "assert args.binding_pins > 0 and args.launch_pins == args.binding_pins + 5\n"
       "for key, value in vars(args).items():\n"
       "    if key.endswith('_sha256'): assert re.fullmatch(r'[0-9a-f]{64}', value), key")
for old, new in [
    ('0ae9699fcb2b70ff33948223e9630a0f8692751baf2fdb27bbb191e7d76d5480', '2dd758b45e2e64294c275a969d05acc6cbeae74ddae509c63c2fa8e40f9e72ee'),
    ('e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121', 'db2a997d44634cb59d7d4bd92408fcf361323d143c6109024b457e5bd4a458de'),
    ('b95fa1160498af88cf3fe38c80aaf12b8707ac3d18f25ed69fdfe54c9920d16f', '647de5513de061fbf231aad4d1d6e728d1b606e38252f946f11f39cfacc7fe36'),
    ('427a91b32475e7f59212954d109865227f6b523c3e43ea9037c14462300a50ff', '3656f44031d9362fdbabeab69d2423266839b88662cbe46b79953a977dc56cd7'),
    ('a6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3', '18c66ebdb2af7138858c08d012c5d3cc7819c1bb2bb5a5b523e488c35e4ed6a2'),
    ('c70e90adab31efd2abcebd9817bf3c3c1ecc84cde546e080f6367c710633423a', '8a4c394b8836dd763d6eb1c5beac73bf49921bcd59fc5f04850b377d0e843044'),
    ('395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d', '825f86468fcc83913066a9a818f4f438d080e054e17aa4aaadc1451974b8688e'),
    ('0899cdf7b9d56481ab71ea4d9c61f86d1104c4d5238d8df4d7e1c5e4e5f7d7a3', '24b9ba22d572874a5e4bd4c97a282e9444126231807ff70d526312303eb12122')]:
    change(old, new)
change('== 71000', '== 81000', 2)
change('ordinary_final_step=71000', 'ordinary_final_step=81000')
change('[1323, 256, 256, 23]', '[1323, 512, 512, 23]')
change("len(source['source_sha256'])==37", "len(source['source_sha256'])==38")
change('direct_target_response_balanced_fit_independent_v3/owner_completion.json',
       'direct_target_width512_fit_independent_v1/owner_completion.json')
change("'2a56cf0ab008f866625a336c8d6445fec2cebcfb4d55f26e8cf3995c0fa33f2c'", 'args.witness_binding_sha256')
change("wb = read(wbp)", "wb = read(wbp)\nassert args.binding_pins > len(wb['input_files'])")
change("wo['pins_exact'] == 5231", "wo['pins_exact'] == len(read(wfolder / 'launch_receipt.json')['input_hashes'])")
change("'75ce87dbc5ffcbf83ffdaff719be5194bf1c4fe2b76a7dfc5b9ebd2eeb07255c'", 'args.witness_launch_sha256')
change("'20268e368ad0dc9453a355f28288b85d8f495b3ee00e3f156161aa7dd3d00b46'", 'args.witness_review_sha256')
change("reviewer='review_continuation'", "reviewer='expert_resume'")
change("features=1323, selected_single_run=True", "features=1323, architecture=[1323,512,512,23], selected_single_run=True")
change("completed_witness_verified=True,", "completed_witness_verified=True,\n"
       "    witness_binding_sha256=args.witness_binding_sha256, witness_launch_sha256=args.witness_launch_sha256,\n"
       "    witness_review_sha256=args.witness_review_sha256,")
tree = ast.parse(source)
# Pure AST validation: parent numerical target arithmetic remains byte-for-byte.
start = "with np.load(wp, allow_pickle=False) as archive:"
end = "driver = (BASE / 'source_draft_v1/evaluate_direct_target_student.py').read_text()"
assert original[original.index(start):original.index(end)] == source[source.index(start):source.index(end)]
for token in ["requested_main_controls=1569", "conditional_hold_controls=250", "maximum_native_steps=18190",
              "expected_head_calls'] == wreport['attempted_head_calls'] == wreport['returned_head_calls'] == 1",
              "for path, digest in pins.items(): assert sha(path) == digest, path",
              "assert not path.exists(), path", "raw_python_exit_preserved'] is True"]:
    assert token in source, token
for forbidden in ['onnxruntime', 'torch', 'mujoco']:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert forbidden not in ast.unparse(node), ast.unparse(node)
with (HERE / 'review_concrete.py').open('x', encoding='utf-8', newline='\n') as f:
    f.write(source)
with (HERE / 'source_delta.patch').open('x', encoding='utf-8', newline='\n') as f:
    f.write(''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
        fromfile='qualified71000/review_concrete.py', tofile='width81000/review_concrete.py')))
with (HERE / 'preparation.json').open('x', encoding='utf-8') as f:
    json.dump(dict(source_preparation_passed=True, actual_checker_executed=False,
        source_sha256=sha(HERE/'review_concrete.py'), parent_checker_sha256=sha(OLD/'review_concrete.py'),
        parent_review_sha256=sha(OLD/'review.json'), derivation_sha256=sha(HERE/'source_delta.patch'),
        generator_sha256=sha(__file__), changes=changes, ast_valid=True,
        target_arithmetic_unchanged=True, full_scope_and_failure_guards_retained=True,
        required_concrete_parameters=['binding_sha256','launch_sha256','witness_owner_sha256','witness_sha256',
            'witness_binding_sha256','witness_launch_sha256','witness_review_sha256','binding_pins','launch_pins'],
        requires_actual_completed_witness=True, requires_root_selection_before_execution=True,
        requested_main_controls=1569, conditional_hold_controls=250,
        task_model_calls=0, ORT_calls=0, native_steps=0, dispatch_performed=False), f, indent=2)
print(json.dumps(dict(checker_sha256=sha(HERE/'review_concrete.py'), preparation_sha256=sha(HERE/'preparation.json'))))
