"""Independent saved-source checks. Does not import the candidate or MuJoCo."""
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'independent_native_stepper_equivalence_v1'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source = BASE / 'source_draft_v1'
    derivation = json.loads((BASE / 'source_derivation.json').read_text())
    checks = []
    for name, item in derivation['copies'].items():
        original = Path(item['original'])
        assert sha(original) == item['sha256'] == sha(source / name), name
        checks.append('exact reviewed v3 copy: ' + name)
    original = Path(derivation['loader_original']['path'])
    assert sha(original) == derivation['loader_original']['sha256']
    original_function = next(n for n in ast.parse(original.read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name == 'load_native_bundle')
    derived_function = next(n for n in ast.parse((source / 'native_loader.py').read_text()).body
                            if isinstance(n, ast.FunctionDef) and n.name == 'load_model')
    stop = next(i for i, n in enumerate(original_function.body)
                if isinstance(n, ast.With) and 'np.load(motion_path' in ast.unparse(n))
    assert len(derived_function.body) == stop + 1
    assert ast.dump(original_function.args) == ast.dump(derived_function.args)
    for a, b in zip(original_function.body[:stop], derived_function.body[:-1]):
        assert ast.dump(a) == ast.dump(b)
    assert ast.unparse(derived_function.body[-1]) == 'return (model, contract)'
    checks.append('model preparation AST exactly original prefix, return model/contract only')
    xml = source / 'root_stub_tests_v1.xml'
    suites = ET.parse(xml).getroot()
    cases = list(suites.iter('testcase'))
    assert len(cases) >= 24
    assert not list(suites.iter('failure')) and not list(suites.iter('error'))
    checks.append(str(len(cases)) + ' root stub tests passed')
    result = dict(saved_source_checks_passed=True, checks=checks,
                  pins={str(p): sha(p) for p in sorted(source.glob('*.py'))},
                  source_derivation_sha256=sha(BASE / 'source_derivation.json'),
                  root_tests_sha256=sha(xml), native_steps=0,
                  model_constructions=0, serializations=0, inference_calls=0,
                  actual_execution_cleared=False)
    target = ROOT / 'source_checks.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(report=str(target), sha256=sha(target), checks=len(checks))))

if __name__ == '__main__':
    main()
