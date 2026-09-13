"""Read-only source/hash/AST review; synthetic results only, no task arrays."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
NEW = BASE.parent
TARGET = NEW / 'direct_target_causal_response_evaluation_v1'
OLD = NEW / 'direct_target_causal_context_evaluation_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    prep_path = TARGET / 'source_preparation.json'
    assert sha(prep_path) == '2195d4f5d06b6c4dec5d2a24038a179be0e266483fc0454edea540705d40fef3'
    prep = read(prep_path)
    current = {p.relative_to(TARGET / 'source_draft_v1').as_posix(): sha(p)
               for p in (TARGET / 'source_draft_v1').rglob('*.py')}
    assert current == prep['source_sha256'] and len(current) == 37
    old_prep = read(OLD / 'source_preparation.json')
    assert sha(OLD / 'source_preparation.json') == prep['original_evaluation_preparation']['sha256']
    unchanged = prep['unchanged_original_files']
    assert len(unchanged) == 33
    for name in unchanged:
        assert current[name] == old_prep['source_sha256'][name] == sha(OLD / 'source_draft_v1' / name)
        assert (TARGET / 'source_draft_v1' / name).read_bytes() == (OLD / 'source_draft_v1' / name).read_bytes()
    old_gate = (OLD / 'source_draft_v1/evaluation_gate.py').read_text()
    replacements = [
        ("binding['ordinary_final_step']==68000", "binding['ordinary_final_step']==71000"),
        ("binding['controller']=='direct_absolute_target_1323_causal_context_study'",
         "binding['controller']=='direct_absolute_target_1323_causal_response_balanced'"),
        ("binding['context_condition'] in ('blinded','causal')", "binding['context_condition']=='causal'"),
        ("'paired_report','shared_manifest','context_alignment','blinded_fit_report','causal_fit_report'",
         "'shared_manifest','context_alignment','energy_source'")]
    expected = old_gate
    for before, after in replacements:
        assert expected.count(before) == 1
        expected = expected.replace(before, after)
    actual = (TARGET / 'source_draft_v1/evaluation_gate.py').read_text()
    assert expected == actual
    assert (TARGET / 'source_draft_v1/context_release.py').read_bytes() == (TARGET / 'context_release_new.py.txt').read_bytes()
    balance = NEW / 'direct_target_response_balance_math_v1/source_prepared_v1/balance_contract.py'
    assert current['balance_contract.py'] == sha(balance) == '3bed156c9a15f2fe22eccd022330c745f7073481bcd36585e6c7b00abce76567'
    diff = ''.join(''.join(difflib.unified_diff(
        (OLD / 'source_draft_v1' / name).read_text().splitlines(True),
        (TARGET / 'source_draft_v1' / name).read_text().splitlines(True),
        fromfile='old/' + name, tofile='new/' + name))
        for name in prep['changed_original_files'])
    assert diff == (TARGET / 'runtime_changes.diff').read_text()
    release_ast = ast.parse((TARGET / 'source_draft_v1/context_release.py').read_text())
    roles = next(ast.literal_eval(n.value) for n in release_ast.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SUBJECTS' for t in n.targets))
    assert len(roles) == len(set(roles)) == 16
    assert roles == ('fit_report','checkpoint','head','normalization','training_manifest','training_request','export_manifest',
                     'coefficient','source_checkpoint','full_state_generation_request','full_state_generation_report',
                     'full_state_data_audit','full_state_data_owner','shared_manifest','context_alignment','energy_source')
    test_path = BASE / 'synthetic_tests_v2.xml'
    suites = list(ET.parse(test_path).getroot())
    assert sum(int(s.attrib['tests']) for s in suites) == 83
    assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('failures','errors','skipped'))
    pins = {str((TARGET / 'source_draft_v1' / name).resolve().as_posix()): value for name, value in current.items()}
    for name in unchanged:
        p = OLD / 'source_draft_v1' / name
        pins[p.as_posix()] = sha(p)
    for p in [prep_path, OLD / 'source_preparation.json', balance, test_path, Path(__file__),
              TARGET / 'prepare_runtime.py', TARGET / 'runtime_derivation.json', TARGET / 'runtime_changes.diff',
              TARGET / 'context_release_new.py.txt', TARGET / 'runtime_tests_v1.xml']:
        pins[p.resolve().as_posix()] = sha(p)
    assert all(sha(p) == value for p, value in pins.items())
    result = dict(passed=True, source_review_pass=True, preparation_only=True, findings=[],
                  source_preparation={'path': prep_path.as_posix(), 'sha256': sha(prep_path)},
                  source_sha256=current, unchanged_original_files=unchanged,
                  exact_gate_substitutions=4, literal_release_subjects=list(roles),
                  ordinary_final_step=71000, optimizer_final_step=6000, condition='causal',
                  focused_scope=['exact source derivation', 'warm continuation/reused normalization and context',
                                 'fixed response group weights and coefficient', 'five exact backend counters',
                                 'direct owner/release and consumed root-audit subjects',
                                 'unchanged native, feature, prior/history and witness runtime'],
                  tests={'passed':83,'failed':0,'skipped':0,'receipt_sha256':sha(test_path)},
                  initial_synthetic_invocation={'tests_run':0,'error':'pytest collection escaped to E:/WpSystem and failed WinError1337',
                                               'resolution':'Reran same synthetic files with explicit source workdir/rootdir/confcutdir; no producer changes.'},
                  input_sha256=pins, model_calls=0, native_steps=0, optimizer_updates=0, actual_task_arrays_read=0,
                  actual_binding_reviewed=False, controller_selected=False, execution_cleared=False,
                  limitations=['Future actual fit, root saved evidence audit, owner and release receipts remain required.',
                               'One concrete witness and canonical launch require their own review/selection.',
                               'Source/synthetic pass does not qualify behavior or deadlines.'])
    with (BASE / 'review.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'passed':True,'sha256':sha(BASE/'review.json'),'pins':len(pins),'tests':83}))


if __name__ == '__main__':
    main()
