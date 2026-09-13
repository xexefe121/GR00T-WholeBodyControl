"""Execute only the actual isolated dict-expression AST on synthetic metadata."""
import ast,copy
from pathlib import Path
import pytest
BASE=Path(__file__).resolve().parent

def expression(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
           and node.func.id=='dict' and any(key.arg=='initialization_sha256' for key in node.keywords)]
    assert len(calls)==1
    return compile(ast.Expression(calls[0]),str(path),'eval')

@pytest.mark.parametrize('preexisting',[{'condition':'causal'},
    {'condition':'causal','initialization_sha256':'old','shared_manifest_sha256':'old'},
    {'condition':'causal','source_receipt_sha256':'old','clearance_sha256':'old'}])
def test_actual_new_request_expression_overrides_duplicate_fields_safely(preexisting):
    request=dict(preexisting,kind='synthetic');before=copy.deepcopy(request)
    values=dict(request=request,condition='causal',dest=Path('synthetic_fit'),shared=Path('synthetic_shared'),
                identity={'frozen':'a'*64,'clearance':'b'*64},sha=lambda path:'c'*64)
    actual=eval(expression(BASE/'source_prepared_v1/train_response_balanced.py'),values)
    assert request==before and actual['condition']=='causal' and actual['kind']=='synthetic'
    assert actual['initialization_sha256']==actual['shared_manifest_sha256']=='c'*64
    assert actual['source_receipt_sha256']=='a'*64 and actual['clearance_sha256']=='b'*64
    old=BASE.parent/'direct_target_causal_response_balanced_student_v1/source_prepared_v1/train_response_balanced.py'
    with pytest.raises(TypeError,match='multiple values'):eval(expression(old),values)
