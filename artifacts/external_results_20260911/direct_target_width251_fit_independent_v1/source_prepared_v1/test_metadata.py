"""Execute isolated auditor metadata AST only; no task arrays or checkpoints."""
import ast
from pathlib import Path
import pytest

SOURCE=Path(__file__).with_name('audit_saved_warm.py')

def tree():return ast.parse(SOURCE.read_text())

def carried_expression():
    matches=[n.args[-1] for n in ast.walk(tree()) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='compare'
        and n.args and isinstance(n.args[0],ast.BinOp)
        and isinstance(n.args[0].right,ast.Constant) and n.args[0].right.value=='_carried_request']
    assert len(matches)==1
    return matches[0]

def test_existing_request_metadata_overridden_without_duplicate_keywords():
    request=dict(condition='causal',initialization_sha256='old-init',shared_manifest_sha256='old-shared',
        source_receipt_sha256='old-receipt',clearance_sha256='old-clearance',updates=3000)
    before=request.copy()
    env=dict(request=request,condition='causal',folder=Path('fit'),shared=Path('fit/shared'),base=Path('base'),
        frozen_sha='new-receipt',sha=lambda path:'hash:'+path.as_posix())
    actual=eval(compile(ast.Expression(carried_expression()),'<isolated carried request>','eval'),env)
    assert request==before
    assert actual==dict(condition='causal',updates=3000,initialization_sha256='hash:fit/initialization.pt',
        shared_manifest_sha256='hash:fit/shared/output_manifest.json',source_receipt_sha256='new-receipt',
        clearance_sha256='hash:base/training_clearance.json')

def test_old_duplicate_keyword_expression_reproduces_failure():
    with pytest.raises(TypeError):dict(**{'condition':'causal'},condition='causal')

def concrete_literal_loop():
    matches=[n for n in ast.walk(tree()) if isinstance(n,ast.For)
        and ast.unparse(n.target)=='(field, digest)' and any(isinstance(v,ast.Constant)
            and v.value=='concrete_literal:' for v in ast.walk(n))]
    assert len(matches)==1
    return matches[0]

def execute_review(review):
    checks=[]
    def check(name,value):
        checks.append((name,value))
        if not value:raise AssertionError(name)
    env=dict(launch_review=review,training_sha='request',frozen_sha='frozen',
        clearance={'launcher_sha256':'launcher'},check=check)
    exec(compile(ast.Module(body=[concrete_literal_loop()],type_ignores=[]),'<literal concrete review>','exec'),env)
    return checks

def test_actual_top_level_concrete_schema_requires_no_subjects_map():
    review=dict(training_request_sha256='request',frozen_receipt_sha256='frozen',launcher_sha256='launcher')
    assert len(execute_review(review))==3 and 'subjects' not in review

@pytest.mark.parametrize('field',['training_request_sha256','frozen_receipt_sha256','launcher_sha256'])
def test_each_concrete_identity_corruption_rejected(field):
    review=dict(training_request_sha256='request',frozen_receipt_sha256='frozen',launcher_sha256='launcher')
    review[field]='changed'
    with pytest.raises(AssertionError):execute_review(review)

def test_nested_unrelated_digest_cannot_replace_literal_subject():
    with pytest.raises(KeyError):execute_review({'subjects':{'request':'request','frozen':'frozen','launcher':'launcher'}})

def test_request_writer_targets_fresh_study_only():
    helper=SOURCE.parent.parent/'prepare_audit_request.py'
    t=ast.parse(helper.read_text())
    value=next(n.value for n in t.body if isinstance(n,ast.Assign)
        and any(isinstance(a,ast.Name) and a.id=='EXPERIMENT' for a in n.targets))
    assert isinstance(value,ast.BinOp) and value.right.value=='direct_target_causal_width512_student_v1'
    assert 'optimization_completed' in helper.read_text() and 'final_export_diagnostics_completed' in helper.read_text()
